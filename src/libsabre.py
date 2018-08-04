import argparse
import json
import math
import sys
import string
from collections import namedtuple
from enum import Enum

class sabre():
    def __init__(self):
        self.reset()

    def reset(self):
        self.manifest = None
        self.buffer_contents = []
        self.buffer_fcc = 0
        self.rebuffer_event_count = 0
        self.rebuffer_time = 0
        self.played_utility = 0
        self.played_bitrate = 0
        self.total_play_time = 0
        self.total_bitrate_change = 0
        self.total_log_bitrate_change = 0
        self.last_played = None

        self.rampup_origin = 0
        self.rampup_time = None
        self.rampup_threshold = 0

        self.sustainable_quality = None
        self.verbose = 0
        self.pending_quality_up = []
        self.max_buffer_size = 0
        self.total_reaction_time = 0

        self.log_history = []
        self.throughput = 0
        self.throughput_history = None

        self.network_total_time = 0

    def load_json(self,path):
        with open(path) as file:
            obj = json.load(file)
        return obj

    def process(self, abr, video='../mmsys18/bbb.json', trace='../mmsys18/4Glogs/report_bus_0002.json', argv = ''):
        
        #self.throughput_history

        parser = argparse.ArgumentParser(description='Simulate an ABR session.',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-f', '--file', metavar='SAVEFILE', default='save.log',
                            help='Save the secific metrics to file.')
        parser.add_argument('-nm', '--network-multiplier', metavar='MULTIPLIER',
                            type=float, default=1,
                            help='Multiply throughput by MULTIPLIER.')
        parser.add_argument('-ml', '--movie-length', metavar='LEN', type=float, default=None,
                            help='Specify the movie length in seconds (use MOVIE length if None).')
        parser.add_argument('-ab', '--abr-basic', action='store_true',
                            help='Set ABR to BASIC (ABR strategy dependant).')
        parser.add_argument('-ao', '--abr-osc', action='store_true',
                            help='Set ABR to minimize oscillations.')
        parser.add_argument('-gp', '--gamma-p', metavar='GAMMAP', type=float, default=5,
                            help='Specify the (gamma p) product in seconds.')
        parser.add_argument('-noibr', '--no-insufficient-buffer-rule', action='store_true',
                            help='Disable Insufficient Buffer Rule.')
        parser.add_argument('-ma', '--moving-average', metavar='AVERAGE',
                            choices=average_list.keys(), default=average_default,
                            help='Specify the moving average strategy (%s).' %
                            ', '.join(average_list.keys()))
        parser.add_argument('-ws', '--window-size', metavar='WINDOW_SIZE',
                            nargs='+', type=int, default=[3],
                            help='Specify sliding window size.')
        parser.add_argument('-hl', '--half-life', metavar='HALF_LIFE',
                            nargs='+', type=float, default=[3, 8],
                            help='Specify EWMA half life.')
        parser.add_argument('-s', '--seek', nargs=2, metavar=('WHEN', 'SEEK'),
                            type=float, default=None,
                            help='Specify when to seek in seconds and where to seek in seconds.')
        choices = ['none', 'left', 'right']
        parser.add_argument('-r', '--replace', metavar='REPLACEMENT',
                            choices=choices, default='none',
                            help='Set replacement strategy (%s).' % ', '.join(choices))
        parser.add_argument('-b', '--max-buffer', metavar='MAXBUFFER', type=float, default=25,
                            help='Specify the maximum buffer size in seconds.')
        parser.add_argument('-noa', '--no-abandon', action='store_true',
                            help='Disable abandonment.')
        parser.add_argument('-rmp', '--rampup-threshold', metavar='THRESHOLD',
                            type=int, default=None,
                            help='Specify at what quality index we are ramped up (None matches network).')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Run in verbose mode.')
        args = parser.parse_args(argv.split(' '))
        args.abr = abr
        args.movie = video
        args.network = trace
        self.verbose = args.verbose

        self.buffer_contents = []
        self.buffer_fcc = 0
        self.pending_quality_up = []
        self.reaction_metrics = []
        self.log_history = []

        self.rebuffer_event_count = 0
        self.rebuffer_time = 0
        self.played_utility = 0
        self.played_bitrate = 0
        self.total_play_time = 0
        self.total_bitrate_change = 0
        self.total_log_bitrate_change = 0
        self.total_reaction_time = 0
        self.last_played = None

        overestimate_count = 0
        overestimate_average = 0
        goodestimate_count = 0
        goodestimate_average = 0
        estimate_average = 0

        self.rampup_origin = 0
        self.rampup_time = None
        self.rampup_threshold = args.rampup_threshold

        self.max_buffer_size = args.max_buffer * 1000
        self.manifest = self.load_json(args.movie)
        bitrates = self.manifest['bitrates_kbps']
        utility_offset = 0 - math.log(bitrates[0])  # so utilities[0] = 0
        utilities = [math.log(b) + utility_offset for b in bitrates]
        if args.movie_length != None:
            l1 = len(self.manifest['segment_sizes_bits'])
            l2 = math.ceil(args.movie_length * 1000 /
                            self.manifest['segment_duration_ms'])
            self.manifest['segment_sizes_bits'] *= math.ceil(l2 / l1)
            self.manifest['segment_sizes_bits'] = self.manifest['segment_sizes_bits'][0:l2]
        manifest = sabre.ManifestInfo(segment_time=self.manifest['segment_duration_ms'],
                                bitrates=bitrates,
                                utilities=utilities,
                                segments=self.manifest['segment_sizes_bits'])

        network_trace = self.load_json(args.network)
        network_trace = [sabre.NetworkPeriod(time=p['duration_ms'],
                                    bandwidth=p['bandwidth_kbps'] *
                                    args.network_multiplier,
                                    latency=p['latency_ms'])
                        for p in network_trace]

        buffer_size = args.max_buffer * 1000
        gamma_p = args.gamma_p

        config = {'buffer_size': buffer_size,
                'gp': gamma_p,
                'abr_osc': args.abr_osc,
                'abr_basic': args.abr_basic,
                'no_ibr': args.no_insufficient_buffer_rule}
        args.abr.use_abr_o = args.abr_osc
        args.abr.use_abr_u = not args.abr_osc
        abr = args.abr(config)
        network = NetworkModel(network_trace,self)

        replacer = NoReplace()

        config = {'window_size': args.window_size, 'half_life': args.half_life}
        throughput_history = average_list[args.moving_average](config)

        # download first segment
        quality = abr.get_first_quality()
        size = manifest.segments[0][quality]
        download_metric = network.download(size, 0, quality, 0)
        download_time = download_metric.time - download_metric.time_to_first_bit
        startup_time = download_time
        self.buffer_contents.append(download_metric.quality)
        t = download_metric.size / download_time
        l = download_metric.time_to_first_bit
        throughput_history.push(download_time, t, l)
        self.log_history.append((download_time,t,l,quality))
        #print('%d,%d -> %d,%d' % (t, l, throughput, latency))
        self.total_play_time += download_metric.time

        # download rest of segments
        next_segment = 1
        abandoned_to_quality = None
        while next_segment < len(manifest.segments):

            full_delay = self.get_buffer_level() + manifest.segment_time - buffer_size
            if full_delay > 0:
                self.deplete_buffer(full_delay)
                network.delay(full_delay)
                abr.report_delay(full_delay)

            if abandoned_to_quality == None:
                (quality, delay) = abr.get_quality_delay(next_segment)
                replace = replacer.check_replace(quality)
            else:
                (quality, delay) = (abandoned_to_quality, 0)
                replace = None
                abandon_to_quality = None

            if replace != None:
                delay = 0
                current_segment = next_segment + replace
                check_abandon = replacer.check_abandon
            else:
                current_segment = next_segment
                check_abandon = abr.check_abandon
            if args.no_abandon:
                check_abandon = None

            size = manifest.segments[current_segment][quality]

            if delay > 0:
                self.deplete_buffer(delay)
                network.delay(delay)

            download_metric = network.download(size, current_segment, quality,
                                            self.get_buffer_level(), check_abandon)

            self.deplete_buffer(download_metric.time)

            # update buffer with new download
            if replace == None:
                if download_metric.abandon_to_quality == None:
                    self.buffer_contents += [quality]
                    next_segment += 1
                else:
                    abandon_to_quality = download_metric.abandon_to_quality
            else:
                # abandon_to_quality == None
                if download_metric.abandon_to_quality == None:
                    if self.get_buffer_level() + manifest.segment_time * replace >= 0:
                        self.buffer_contents[replace] = quality
                    else:
                        print('WARNING: too late to replace')
                        pass
                else:
                    pass


            abr.report_download(download_metric, replace != None)

            # calculate throughput and latency
            download_time = download_metric.time - download_metric.time_to_first_bit
            t = download_metric.downloaded / download_time
            l = download_metric.time_to_first_bit

            # check accuracy of throughput estimate
            if self.throughput > t:
                overestimate_count += 1
                overestimate_average += (self.throughput - t -
                                        overestimate_average) / overestimate_count
            else:
                goodestimate_count += 1
                goodestimate_average += (t - self.throughput -
                                        goodestimate_average) / goodestimate_count
            estimate_average += ((self.throughput - t - estimate_average) /
                                (overestimate_count + goodestimate_count))

            # update throughput estimate
            if download_metric.abandon_to_quality == None:
                self.log_history.append((download_time,t,l))
                self.throughput_history.push(download_time, t, l)

        self.playout_buffer()
        return self.played_bitrate, self.rebuffer_time, self.total_bitrate_change
        
    ManifestInfo = namedtuple(
        'ManifestInfo', 'segment_time bitrates utilities segments')
    NetworkPeriod = namedtuple('NetworkPeriod', 'time bandwidth latency')

    DownloadProgress = namedtuple('DownloadProgress',
                                'index quality '
                                'size downloaded '
                                'time time_to_first_bit '
                                'abandon_to_quality')


    def get_buffer_level(self):
        return self.manifest.segment_time * len(self.buffer_contents) - self.buffer_fcc
    
    def deplete_buffer(self,time):

        if len(self.buffer_contents) == 0:
            self.rebuffer_time += time
            self.total_play_time += time
            return

        if self.buffer_fcc > 0:
            # first play any partial chunk left

            if time + self.buffer_fcc < self.manifest.segment_time:
                self.buffer_fcc += time
                self.total_play_time += time
                return

            time -= self.manifest.segment_time - self.buffer_fcc
            self.total_play_time += self.manifest.segment_time - self.buffer_fcc
            self.buffer_contents.pop(0)
            self.buffer_fcc = 0

        # buffer_fcc == 0 if we're here

        while time > 0 and len(self.buffer_contents) > 0:

            quality = self.buffer_contents[0]
            self.played_utility += self.manifest.utilities[quality]
            self.played_bitrate += self.manifest.bitrates[quality]
            if quality != self.last_played and self.last_played != None:
                self.total_bitrate_change += abs(self.manifest.bitrates[quality] -
                                            self.manifest.bitrates[self.last_played])
                self.total_log_bitrate_change += abs(math.log(self.manifest.bitrates[quality] /
                                                        self.manifest.bitrates[self.last_played]))
            self.last_played = quality

            if self.rampup_time == None:
                rt = self.sustainable_quality if self.rampup_threshold == None else self.rampup_threshold
                if quality >= rt:
                    self.rampup_time = self.total_play_time - self.rampup_origin

            # bookkeeping to track reaction time to increased bandwidth
            for p in self.pending_quality_up:
                if len(p) == 2 and quality >= p[1]:
                    p.append(self.total_play_time)

            if time >= self.manifest.segment_time:
                self.buffer_contents.pop(0)
                self.total_play_time += self.manifest.segment_time
                time -= self.manifest.segment_time
            else:
                self.buffer_fcc = time
                self.total_play_time += time
                time = 0

        if time > 0:
            self.rebuffer_time += time
            self.total_play_time += time
            self.rebuffer_event_count += 1

        self.process_quality_up(self.total_play_time)
    
    def playout_buffer(self):
        self.deplete_buffer(self.get_buffer_level())
        del self.buffer_contents[:]
        self.buffer_fcc = 0


    def process_quality_up(self,now):
        # check which switches can be processed

        cutoff = now - self.max_buffer_size
        while len(self.pending_quality_up) > 0 and self.pending_quality_up[0][0] < cutoff:
            p = self.pending_quality_up.pop(0)
            if len(p) == 2:
                reaction = self.max_buffer_size
            else:
                reaction = min(self.max_buffer_size, p[2] - p[0])
            #print('\n[%d] reaction time: %d' % (now, reaction))
            # print(p)
            self.total_reaction_time += reaction


    def advertize_new_network_quality(self, quality, previous_quality):
        # self.max_buffer_size
        # self.network_total_time
        # self.pending_quality_up
        # self.buffer_contents

        # bookkeeping to track reaction time to increased bandwidth

        # process any previous quality up switches that have "matured"
        self.process_quality_up(self.network_total_time)

        # mark any pending switch up done if new quality switches back below its quality
        for p in self.pending_quality_up:
            if len(p) == 2 and p[1] > quality:
                p.append(self.network_total_time)
        #pending_quality_up = [p for p in pending_quality_up if p[1] >= quality]

        # filter out switches which are not upwards (three separate checks)
        if quality <= previous_quality:
            return
        for q in self.buffer_contents:
            if quality <= q:
                return
        for p in self.pending_quality_up:
            if quality <= p[1]:
                return

        # valid quality up switch
        self.pending_quality_up.append([self.network_total_time, quality])


    class NetworkModel:

        min_progress_size = 12000
        min_progress_time = 50

        def __init__(self, network_trace, sabre):
            self.sabre = sabre
            self.sabre.sustainable_quality = None
            self.sabre.network_total_time = 0
            self.trace = network_trace
            self.index = -1
            self.time_to_next = 0
            self.next_network_period()

        def next_network_period(self):
            self.index += 1
            if self.index == len(self.trace):
                self.index = 0
            self.time_to_next = self.trace[self.index].time

            latency_factor = 1 - \
                self.trace[self.index].latency / self.sabre.manifest.segment_time
            effective_bandwidth = self.trace[self.index].bandwidth * latency_factor

            previous_sustainable_quality = self.sabre.sustainable_quality
            sustainable_quality = 0
            for i in range(1, len(self.sabre.manifest.bitrates)):
                if self.sabre.manifest.bitrates[i] > effective_bandwidth:
                    break
                sustainable_quality = i
            if (sustainable_quality != previous_sustainable_quality and
                    previous_sustainable_quality != None):
                self.sabre.advertize_new_network_quality(
                    sustainable_quality, previous_sustainable_quality)

        # return delay time
        def do_latency_delay(self, delay_units):
            total_delay = 0
            while delay_units > 0:
                current_latency = self.trace[self.index].latency
                time = delay_units * current_latency
                if time <= self.time_to_next:
                    total_delay += time
                    self.sabre.network_total_time += time
                    self.time_to_next -= time
                    delay_units = 0
                else:
                    # time > self.time_to_next implies current_latency > 0
                    total_delay += self.time_to_next
                    self.sabre.network_total_time += self.time_to_next
                    delay_units -= self.time_to_next / current_latency
                    self.next_network_period()
            return total_delay

        # return download time
        def do_download(self, size):
            total_download_time = 0
            while size > 0:
                current_bandwidth = self.trace[self.index].bandwidth
                if size <= self.time_to_next * current_bandwidth:
                    # current_bandwidth > 0
                    time = size / current_bandwidth
                    total_download_time += time
                    self.sabre.network_total_time += time
                    self.time_to_next -= time
                    size = 0
                else:
                    total_download_time += self.time_to_next
                    self.sabre.network_total_time += self.time_to_next
                    size -= self.time_to_next * current_bandwidth
                    self.next_network_period()
            return total_download_time

        def do_minimal_latency_delay(self, delay_units, min_time):
            total_delay_units = 0
            total_delay_time = 0
            while delay_units > 0 and min_time > 0:
                current_latency = self.trace[self.index].latency
                time = delay_units * current_latency
                if time <= min_time and time <= self.time_to_next:
                    units = delay_units
                    self.time_to_next -= time
                    self.sabre.network_total_time += time
                elif min_time <= self.time_to_next:
                    # time > 0 implies current_latency > 0
                    time = min_time
                    units = time / current_latency
                    self.time_to_next -= time
                    self.sabre.network_total_time += time
                else:
                    time = self.time_to_next
                    units = time / current_latency
                    self.sabre.network_total_time += time
                    self.next_network_period()
                total_delay_units += units
                total_delay_time += time
                delay_units -= units
                min_time -= time
            return (total_delay_units, total_delay_time)

        def do_minimal_download(self, size, min_size, min_time):
            total_size = 0
            total_time = 0
            while size > 0 and (min_size > 0 or min_time > 0):
                current_bandwidth = self.trace[self.index].bandwidth
                if current_bandwidth > 0:
                    min_bits = max(min_size, min_time * current_bandwidth)
                    bits_to_next = self.time_to_next * current_bandwidth
                    if size <= min_bits and size <= bits_to_next:
                        bits = size
                        time = bits / current_bandwidth
                        self.time_to_next -= time
                        self.sabre.network_total_time += time
                    elif min_bits <= bits_to_next:
                        bits = min_bits
                        time = bits / current_bandwidth
                        # make sure rounding error does not push while loop into endless loop
                        min_size = 0
                        min_time = 0
                        self.time_to_next -= time
                        self.sabre.network_total_time += time
                    else:
                        bits = bits_to_next
                        time = self.time_to_next
                        self.sabre.network_total_time += time
                        self.next_network_period()
                else:  # current_bandwidth == 0
                    bits = 0
                    if min_size > 0 or min_time > self.time_to_next:
                        time = self.time_to_next
                        self.sabre.network_total_time += time
                        self.next_network_period()
                    else:
                        time = min_time
                        self.time_to_next -= time
                        self.sabre.network_total_time += time
                total_size += bits
                total_time += time
                size -= bits
                min_size -= bits
                min_time -= time
            return (total_size, total_time)

        def delay(self, time):
            while time > self.time_to_next:
                time -= self.time_to_next
                self.sabre.network_total_time += self.time_to_next
                self.next_network_period()
            self.time_to_next -= time
            self.sabre.network_total_time += time

        def download(self, size, idx, quality, buffer_level, check_abandon=None):
            if size <= 0:
                return sabre.DownloadProgress(index=idx, quality=quality,
                                        size=0, downloaded=0,
                                        time=0, time_to_first_bit=0,
                                        abandon_to_quality=None)

            if not check_abandon or (sabre.NetworkModel.min_progress_time <= 0 and
                                    sabre.NetworkModel.min_progress_size <= 0):
                latency = self.do_latency_delay(1)
                time = latency + self.do_download(size)
                return sabre.DownloadProgress(index=idx, quality=quality,
                                        size=size, downloaded=size,
                                        time=time, time_to_first_bit=latency,
                                        abandon_to_quality=None)

            total_download_time = 0
            total_download_size = 0
            min_time_to_progress = sabre.NetworkModel.min_progress_time
            min_size_to_progress = sabre.NetworkModel.min_progress_size

            if sabre.NetworkModel.min_progress_size > 0:
                latency = self.do_latency_delay(1)
                total_download_time += latency
                min_time_to_progress -= total_download_time
                delay_units = 0
            else:
                latency = None
                delay_units = 1

            abandon_quality = None
            while total_download_size < size and abandon_quality == None:

                if delay_units > 0:
                    # NetworkModel.min_progress_size <= 0
                    (units, time) = self.do_minimal_latency_delay(
                        delay_units, min_time_to_progress)
                    total_download_time += time
                    delay_units -= units
                    min_time_to_progress -= time
                    if delay_units <= 0:
                        latency = total_download_time

                if delay_units <= 0:
                    # don't use else to allow fall through
                    (bits, time) = self.do_minimal_download(size - total_download_size,
                                                            min_size_to_progress, min_time_to_progress)
                    total_download_time += time
                    total_download_size += bits
                    # no need to upldate min_[time|size]_to_progress - reset below

                dp = sabre.DownloadProgress(index=idx, quality=quality,
                                    size=size, downloaded=total_download_size,
                                    time=total_download_time, time_to_first_bit=latency,
                                    abandon_to_quality=None)
                if total_download_size < size:
                    abandon_quality = check_abandon(
                        dp, max(0, buffer_level - total_download_time))
                    min_time_to_progress = sabre.NetworkModel.min_progress_time
                    min_size_to_progress = sabre.NetworkModel.min_progress_size

            return sabre.DownloadProgress(index=idx, quality=quality,
                                    size=size, downloaded=total_download_size,
                                    time=total_download_time, time_to_first_bit=latency,
                                    abandon_to_quality=abandon_quality)

class Abr:

    def __init__(self, config):
        pass

    def get_quality_delay(self, segment_index):
        raise NotImplementedError

    def get_first_quality(self):
        return 0

    def report_delay(self, delay):
        pass

    def report_download(self, metrics, is_replacment):
        pass

    def report_seek(self, where):
        pass

    def check_abandon(self, progress, buffer_level):
        return None

    def quality_from_throughput(self, tput):
        global manifest
        global throughput
        global latency

        p = manifest.segment_time

        quality = 0
        while (quality + 1 < len(manifest.bitrates) and
               latency + p * manifest.bitrates[quality + 1] / tput <= p):
            quality += 1
        return quality