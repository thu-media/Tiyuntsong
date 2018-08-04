import var

class Abr:
    safety_factor = 0.9
    def __init__(self):
        if var.super_verbose:
            print('[info] ABR.init')

    # def get_quality_delay(self):
        # raise NotImplementedError
    def get_first_quality(self):
        if var.super_verbose:
            print('[info] ABR.get_first_quality')
        return 0
    def report_delay(self, delay):
        pass
    def report_download(self, metrics, is_replacment):
        pass
    def quality_from_throughput(self, tput):
        if var.super_verbose:
            print('[info] ABR.quality_from_throughput')
        p = var.manifest.segment_time

        quality = 0
        while (quality + 1 < len(var.manifest.bitrates) and
               var.latency + p * var.manifest.bitrates[quality + 1] / (tput*self.safety_factor) <= p):
            quality += 1
        return quality
