import json, math
from collections import namedtuple
import var

ManifestInfo = namedtuple('ManifestInfo', 'segment_time bitrates utilities segments')
NetworkPeriod = namedtuple('NetworkPeriod', 'time bandwidth latency')
DownloadProgress = namedtuple('DownloadProgress',
                              'index quality '
                              'size downloaded '
                              'time time_to_first_bit ')

def load_json(path):
    with open(path) as file:
        obj = json.load(file)
    return obj

def get_buffer_level():
    if var.super_verbose:
        print('[info] get_buffer_level')
    return var.manifest.segment_time * len(var.buffer_contents) - var.buffer_fcc

def deplete_buffer(time):
    if var.super_verbose:
        print('[info] deplete_buffer')
    if len(var.buffer_contents) == 0:
        var.rebuffer_time += time
        var.total_play_time += time
        return

    if var.buffer_fcc > 0:
        # first play any partial chunk left

        if time + var.buffer_fcc < var.manifest.segment_time:
            var.buffer_fcc += time
            var.total_play_time += time
            return

        time -= var.manifest.segment_time - var.buffer_fcc
        var.total_play_time += var.manifest.segment_time - var.buffer_fcc
        var.buffer_contents.pop(0)
        var.buffer_fcc = 0

    # var.buffer_fcc == 0 if we're here
    while time > 0 and len(var.buffer_contents) > 0:
        quality = var.buffer_contents[0]
        var.played_utility += var.manifest.utilities[quality]
        var.played_bitrate += var.manifest.bitrates[quality]
        if quality != var.last_played and var.last_played != None:
            var.total_bitrate_change += abs(var.manifest.bitrates[quality] -
                                        var.manifest.bitrates[var.last_played])
            var.total_log_bitrate_change += abs(math.log(float(var.manifest.bitrates[quality]) /
                                                     var.manifest.bitrates[var.last_played]))
        var.last_played = quality

        if var.rampup_time == None:
            rt = var.sustainable_quality #if rampup_threshold == None else rampup_threshold
            if quality >= rt:
                var.rampup_time = var.total_play_time - var.rampup_origin

        # bookkeeping to track reaction time to increased bandwidth
        for p in var.pending_quality_up:
            if len(p) == 2 and quality >= p[1]:
                p.append(var.total_play_time)

        if time >= var.manifest.segment_time:
            var.buffer_contents.pop(0)
            var.total_play_time += var.manifest.segment_time
            time -= var.manifest.segment_time
        else:
            var.buffer_fcc = time
            var.total_play_time += time
            time = 0

    if time > 0:
        var.rebuffer_time += time
        var.total_play_time += time
        var.rebuffer_event_count += 1

    process_quality_up(var.total_play_time)

def playout_buffer():
    if var.super_verbose:
        print('[info] playout_buffer')
    deplete_buffer(get_buffer_level())

    # make sure no rounding error
    del var.buffer_contents[:]
    var.buffer_fcc = 0

def process_quality_up(now):
    if var.super_verbose:
        print('[info] process_quality_up')
    # check which switches can be processed

    cutoff = now - var.max_buffer_size
    while len(var.pending_quality_up) > 0 and var.pending_quality_up[0][0] < cutoff:
        p = var.pending_quality_up.pop(0)
        if len(p) == 2:
            reaction = var.max_buffer_size
        else:
            reaction = min(var.max_buffer_size, p[2] - p[0])
        #print('\n[%d] reaction time: %d' % (now, reaction))
        #print(p)
        var.total_reaction_time += reaction

def advertize_new_network_quality(quality, previous_quality):
    if var.super_verbose:
        print('[info] advertize_new_network_quality')
    # bookkeeping to track reaction time to increased bandwidth
    # process any previous quality up switches that have "matured"
    process_quality_up(var.network_total_time)

    # mark any pending switch up done if new quality switches back below its quality
    for p in var.pending_quality_up:
        if len(p) == 2 and p[1] > quality:
            p.append(var.network_total_time)
    #var.pending_quality_up = [p for p in var.pending_quality_up if p[1] >= quality]

    # filter out switches which are not upwards (three separate checks)
    if quality <= previous_quality:
        return
    for q in var.buffer_contents:
        if quality <= q:
            return
    for p in var.pending_quality_up:
        if quality <= p[1]:
            return

    # valid quality up switch
    var.pending_quality_up.append([var.network_total_time, quality])