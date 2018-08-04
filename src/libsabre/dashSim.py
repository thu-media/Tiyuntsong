import argparse
# import sys
# import string
# from enum import Enum
import var
from func import *
#from thrupt import *
from network import *
from abr import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Simulate an ABR session.',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--network', metavar = 'NETWORK', default = 'network.json',
                        help = 'Specify the .json file describing the network trace.')
    parser.add_argument('-nm', '--network-multiplier', metavar = 'MULTIPLIER',
                        type = float, default = 1,
                        help = 'Multiply var.throughput by MULTIPLIER.')
    parser.add_argument('-m', '--movie', metavar = 'MOVIE', default = 'movie.json',
                        help = 'Specify the .json file describing the movie chunks.')

    # other options
    parser.add_argument('-ws', '--window-size', metavar = 'WINDOW_SIZE',
                        nargs = '+', type = int, default = [3],
                        help = 'Specify sliding window size.')
    parser.add_argument('-b', '--max-buffer', metavar = 'MAXBUFFER', type = float, default = 25,
                        help = 'Specify the maximum buffer size in seconds.')
    parser.add_argument('-v', '--verbose', action = 'store_true',
                        help = 'Run in verbose mode.')
    args = parser.parse_args()

    # global variables shared accross modules
    var.verbose = args.verbose
    var.max_buffer_size = args.max_buffer * 1000
    var.manifest = load_json(args.movie)
    var.buffer_contents = []
    var.buffer_fcc = 0
    var.pending_quality_up = []
    var.rebuffer_event_count = 0
    var.rebuffer_time = 0
    var.played_utility = 0
    var.played_bitrate = 0
    var.total_play_time = 0
    var.total_bitrate_change = 0
    var.total_log_bitrate_change = 0
    var.total_reaction_time = 0
    var.last_played = None
    var.rampup_origin = 0
    var.rampup_time = None
    
    # log the resuls    
    overestimate_count = 0
    overestimate_average = 0
    goodestimate_count = 0
    goodestimate_average = 0
    estimate_average = 0

    # other variables
    bitrates = var.manifest['bitrates_kbps']
    utilities = [math.log(b/bitrates[0]) for b in bitrates]
    var.manifest = ManifestInfo(segment_time = var.manifest['segment_duration_ms'],
                            bitrates     = bitrates,
                            utilities    = utilities,
                            segments     = var.manifest['segment_sizes_bits'])
    network_trace = load_json(args.network)
    network_trace = [NetworkPeriod(time      = p['duration_ms'],
                                   bandwidth = p['bandwidth_kbps'] * args.network_multiplier,
                                   latency   = p['latency_ms'])
                     for p in network_trace]

    buffer_size = args.max_buffer * 1000 # in ms

    abr = Abr()
    network = NetworkModel(network_trace)

    config = {'window_size': args.window_size}
    var.throughput_history = SlidingWindow(config)

    # download first segment
    quality = abr.get_first_quality()
    size = var.manifest.segments[0][quality]
    download_metric = network.download(size, 0, quality, 0)
    download_time = download_metric.time - download_metric.time_to_first_bit
    startup_time = download_time
    var.buffer_contents.append(download_metric.quality)
    t = download_metric.size / download_time
    l = download_metric.time_to_first_bit
    var.throughput_history.push(download_time, t, l)
    #print('%d,%d -> %d,%d' % (t, l, var.throughput, var.latency))
    var.total_play_time += download_metric.time

    if var.verbose:
        print('[log ] [%d-%d]  %d: q=%d s=%d/%d t=%d=%d+%d bl=0->0->%d' %
              (0, round(download_metric.time), 0, download_metric.quality,
               download_metric.downloaded, download_metric.size,
               download_metric.time, download_metric.time_to_first_bit,
               download_metric.time - download_metric.time_to_first_bit,
               get_buffer_level()))

     # download rest of segments
    current_segment = 1
    while current_segment < len(var.manifest.segments):
        #input('===================pause===================\n')
        # do we have space for a new segment on the buffer?
        full_delay = get_buffer_level() + var.manifest.segment_time - buffer_size
        if full_delay > 0:
            deplete_buffer(full_delay)
            network.delay(full_delay)
            abr.report_delay(full_delay)
            if var.verbose:
                print('[log ] full buffer delay %d bl=%d' % (full_delay, get_buffer_level()))
        
        # quality = abr.get_quality_delay()
        quality = abr.quality_from_throughput(var.throughput)

        size = var.manifest.segments[current_segment][quality]

        download_metric = network.download(size, current_segment, quality,
                                           get_buffer_level())

        result = ''
        if var.verbose:
            result += ('[log ] [%d-%d]  %d: q=%d s=%d/%d t=%d=%d+%d ' %
                  (round(var.total_play_time), round(var.total_play_time + download_metric.time),
                   current_segment, download_metric.quality,
                   download_metric.downloaded, download_metric.size,
                   download_metric.time, download_metric.time_to_first_bit,
                   download_metric.time - download_metric.time_to_first_bit))
            result += ('bl=%d' % get_buffer_level())

        deplete_buffer(download_metric.time)
        if var.verbose:
            result += ('->%d' % get_buffer_level())

        # update buffer with new download
        var.buffer_contents += [quality]
        current_segment += 1
        if var.verbose:
            result += ('->%d' % get_buffer_level())
            print(result)

        # abr.report_download(download_metric, replace != None)
        abr.report_download(download_metric, False)

        # calculate var.throughput and var.latency
        download_time = download_metric.time - download_metric.time_to_first_bit
        t = download_metric.downloaded / download_time
        l = download_metric.time_to_first_bit

        # check accuracy of var.throughput estimate
        if var.throughput > t:
            overestimate_count += 1
            overestimate_average += (var.throughput - t - overestimate_average) / overestimate_count
        else:
            goodestimate_count += 1
            goodestimate_average += (t - var.throughput - goodestimate_average) / goodestimate_count
        estimate_average += ((var.throughput - t - estimate_average) /
                             (overestimate_count + goodestimate_count))

        # update var.throughput estimate
        var.throughput_history.push(download_time, t, l)
        # loop while next_segment < len(var.manifest.segments)

    playout_buffer()

    # multiply by to_time_average to get per/chunk average
    to_time_average = 1 / (var.total_play_time / var.manifest.segment_time)
    count = len(var.manifest.segments)
    time = count * var.manifest.segment_time + var.rebuffer_time + startup_time
    print('[log ] buffer size: %d' % buffer_size)
    print('[log ] total played utility: %f' % var.played_utility)
    print('[log ] time average played utility: %f' % (var.played_utility * to_time_average))
    print('[log ] total played bitrate: %f' % var.played_bitrate)
    print('[log ] time average played bitrate: %f' % (var.played_bitrate * to_time_average))
    print('[log ] total play time: %f' % (var.total_play_time / 1000))
    print('[log ] total play time chunks: %f' % (var.total_play_time / var.manifest.segment_time))
    print('[log ] total rebuffer: %f' % (var.rebuffer_time / 1000))
    print('[log ] rebuffer ratio: %f' % (var.rebuffer_time / var.total_play_time))
    print('[log ] time average rebuffer: %f' % (var.rebuffer_time / 1000 * to_time_average))
    print('[log ] total rebuffer events: %f' % var.rebuffer_event_count)
    print('[log ] time average rebuffer events: %f' % (var.rebuffer_event_count * to_time_average))
    print('[log ] total bitrate change: %f' % var.total_bitrate_change)
    print('[log ] time average bitrate change: %f' % (var.total_bitrate_change * to_time_average))
    print('[log ] totallog  bitrate change: %f' % var.total_log_bitrate_change)
    print('[log ] time averagelog  bitrate change: %f' % (var.total_log_bitrate_change * to_time_average))
    print('[log ] time average score: %f' %
          (to_time_average * (var.played_utility -
                              5 * var.rebuffer_time / var.manifest.segment_time)))

    if overestimate_count == 0:
        print('[log ] over estimate count: 0')
        print('[log ] over estimate: 0')
    else:
        print('[log ] over estimate count: %d' % overestimate_count)
        print('[log ] over estimate: %f' % overestimate_average)

    if goodestimate_count == 0:
        print('[log ] leq estimate count: 0')
        print('[log ] leq estimate: 0')
    else:
        print('[log ] leq estimate count: %d' % goodestimate_count)
        print('[log ] leq estimate: %f' % goodestimate_average)
    print('[log ] estimate: %f' % estimate_average)

    if var.rampup_time == None:
        print('[log ] rampup time: %f' % (len(var.manifest.segments) * var.manifest.segment_time / 1000))
    else:
        print('[log ] rampup time: %f' % (var.rampup_time / 1000))

    print('[log ] total reaction time: %f' % (var.total_reaction_time / 1000))
