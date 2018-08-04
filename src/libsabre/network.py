import var, func

class NetworkModel:

    min_progress_size = 12000
    min_progress_time = 50

    def __init__(self, network_trace):
        print('[info] NetworkModel.init')

        var.sustainable_quality = None
        var.network_total_time = 0
        self.trace = network_trace
        self.index = -1
        self.time_to_next = 0
        self.next_network_period()

    def next_network_period(self):
        if var.super_verbose:
            print('[info] NetworkModel.next_network_period')
        self.index += 1
        if self.index == len(self.trace):
            self.index = 0
        self.time_to_next = self.trace[self.index].time

        # latency_factor = 1 - self.trace[self.index].latency / self.trace[self.index].time
        latency_factor = 1 - self.trace[self.index].latency / var.manifest.segment_time # WRONG?
        effective_bandwidth = self.trace[self.index].bandwidth * latency_factor

        previous_sustainable_quality = var.sustainable_quality
        var.sustainable_quality = 0
        for i in range(1, len(var.manifest.bitrates)):
            if var.manifest.bitrates[i] > effective_bandwidth:
                break
            var.sustainable_quality = i
        if (var.sustainable_quality != previous_sustainable_quality and
            previous_sustainable_quality != None):
            func.advertize_new_network_quality(var.sustainable_quality, previous_sustainable_quality)

        if var.verbose:
            print('[log ] [%d] Network: %d kbps, %dms  (q=%d: bitrate=%d kbps)' %
                  (round(var.network_total_time),
                   self.trace[self.index].bandwidth, self.trace[self.index].latency,
                   var.sustainable_quality, var.manifest.bitrates[var.sustainable_quality]))

    # return delay time
    def do_latency_delay(self, delay_units):
        if var.super_verbose:
            print('[info] NetworkModel.do_latency_delay')

        total_delay = 0
        while delay_units > 0:
            current_latency = self.trace[self.index].latency
            time = delay_units * current_latency
            if time <= self.time_to_next:
                total_delay += time
                var.network_total_time += time
                self.time_to_next -= time
                delay_units = 0
            else:
                # time > self.time_to_next implies current_latency > 0
                total_delay += self.time_to_next
                var.network_total_time += self.time_to_next
                delay_units -= self.time_to_next / current_latency
                self.next_network_period()
        return total_delay

    # return download time
    def do_download(self, size):
        if var.super_verbose:
            print('[info] NetworkModel.do_download')
        total_download_time = 0
        while size > 0:
            current_bandwidth = self.trace[self.index].bandwidth
            if size <= self.time_to_next * current_bandwidth:
                # current_bandwidth > 0
                time = size / current_bandwidth
                total_download_time += time
                var.network_total_time += time
                self.time_to_next -= time
                size = 0
            else:
                total_download_time += self.time_to_next
                var.network_total_time += self.time_to_next
                size -= self.time_to_next * current_bandwidth
                self.next_network_period()
        return total_download_time

    def do_minimal_latency_delay(self, delay_units, min_time):
        if var.super_verbose:
            print('[info] NetworkModel.do_minimal_latency_delay')
        total_delay_units = 0
        total_delay_time = 0
        while delay_units > 0 and min_time > 0:
            current_latency = self.trace[self.index].latency
            time = delay_units * current_latency
            if time <= min_time and time <= self.time_to_next:
                units = delay_units
                self.time_to_next -= time
                var.network_total_time += time
            elif min_time <= self.time_to_next:
                # time > 0 implies current_latency > 0
                time = min_time
                units = time / current_latency
                self.time_to_next -= time
                var.network_total_time += time
            else:
                time = self.time_to_next
                units = time / current_latency
                var.network_total_time += time
                self.next_network_period()
            total_delay_units += units
            total_delay_time += time
            delay_units -= units
            min_time -= time
        return (total_delay_units, total_delay_time)

    def do_minimal_download(self, size, min_size, min_time):
        if var.super_verbose:
            print('[info] NetworkModel.do_minial_download')
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
                    var.network_total_time += time
                elif min_bits <= bits_to_next:
                    bits = min_bits
                    time = bits / current_bandwidth
                    # make sure rounding error does not push while loop into endless loop
                    min_size = 0
                    min_time = 0
                    self.time_to_next -= time
                    var.network_total_time += time
                else:
                    bits = bits_to_next
                    time = self.time_to_next
                    var.network_total_time += time
                    self.next_network_period()
            else: # current_bandwidth == 0
                bits = 0
                if min_size > 0 or min_time > self.time_to_next:
                    time = self.time_to_next
                    var.network_total_time += time
                    self.next_network_period()
                else:
                    time = min_time
                    self.time_to_next -= time
                    var.network_total_time += time
            total_size += bits
            total_time += time
            size -= bits
            min_size -= bits
            min_time -= time
        return (total_size, total_time)

    def delay(self, time):
        if var.super_verbose:
            print('[info] NetworkModel.delay')
        while time > self.time_to_next:
            time -= self.time_to_next
            var.network_total_time += self.time_to_next
            self.next_network_period()
        self.time_to_next -= time
        var.network_total_time += time

    def download(self, size, idx, quality, buffer_level):
        if var.super_verbose:
            print('[info] NetworkModel.download')
        if size <= 0:
            return func.DownloadProgress(index = idx, quality = quality,
                                    size = 0, downloaded = 0,
                                    time = 0, time_to_first_bit = 0)

        if NetworkModel.min_progress_time <= 0 and NetworkModel.min_progress_size <= 0:                                
            var.latency = self.do_latency_delay(1)
            time = var.latency + self.do_download(size)
            return func.DownloadProgress(index = idx, quality = quality,
                                    size = size, downloaded = size,
                                    time = time, time_to_first_bit = var.latency) 

        total_download_time = 0
        total_download_size = 0
        min_time_to_progress = NetworkModel.min_progress_time
        min_size_to_progress = NetworkModel.min_progress_size

        if NetworkModel.min_progress_size > 0:
            var.latency = self.do_latency_delay(1)
            total_download_time += var.latency
            min_time_to_progress -= total_download_time
            delay_units = 0
        else:
            var.latency = None
            delay_units = 1

        while total_download_size < size: 

            if delay_units > 0:
                # NetworkModel.min_progress_size <= 0
                (units, time) = self.do_minimal_latency_delay(delay_units, min_time_to_progress)
                total_download_time += time
                delay_units -= units
                min_time_to_progress -= time
                if delay_units <= 0:
                    var.latency = total_download_time

            if delay_units <= 0:
                # don't use else to allow fall through
                (bits, time) = self.do_minimal_download(size - total_download_size,
                                                        min_size_to_progress, min_time_to_progress)
                total_download_time += time
                total_download_size += bits
                # no need to upldate min_[time|size]_to_progress - reset below

            if total_download_size < size:
                min_time_to_progress = NetworkModel.min_progress_time
                min_size_to_progress = NetworkModel.min_progress_size

        return func.DownloadProgress(index = idx, quality = quality,
                                size = size, downloaded = total_download_size,
                                time = total_download_time, time_to_first_bit = var.latency)


class ThroughputHistory:
    def __init__(self, config):
        pass
    def push(self, time, tput, lat):
        raise NotImplementedError

class SlidingWindow(ThroughputHistory):

    default_window_size = [3]
    max_store = 20

    def __init__(self, config):
        if var.super_verbose:
            print('[info] SlidingWindow.init')


        if 'window_size' in config and config['window_size'] != None:
            self.window_size = config['window_size']
        else:
            self.window_size = SlidingWindow.default_window_size

        # TODO: init somewhere else?
        # print(var.keys())
        var.throughput = None
        var.latency = None

        self.last_throughputs = []
        self.last_latencies = []

    def push(self, time, tput, lat):
        if var.super_verbose:
            print('[info] SlidingWindow.push')
        self.last_throughputs += [tput]
        self.last_throughputs = self.last_throughputs[-SlidingWindow.max_store:]

        self.last_latencies += [lat]
        self.last_latencies = self.last_latencies[-SlidingWindow.max_store:]

        tput = None
        lat = None
        for ws in self.window_size:
            sample = self.last_throughputs[-ws:]
            t = sum(sample) / len(sample)
            tput = t if tput == None else min(tput, t) # conservative min
            sample = self.last_latencies[-ws:]
            l = sum(sample) / len(sample)
            lat = l if lat == None else max(lat, l) # conservative max
        var.throughput = tput
        var.latency = lat