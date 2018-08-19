import os
import sys


class log:
    def __init__(self, filename):
        self.filename = filename
        self.log_file = open(self.filename, 'w')

    def write_log(self, agent_result):
        for p in agent_result:
            #print(p)
            total_bitrate0, total_rebuffer0, total_smoothness0 = p
            #total_bitrate1, total_rebuffer1, _ = agent_result[1]
            bitrate = round(total_bitrate0, 2)
            rebuffer = round(total_rebuffer0, 2)
            smoothness = round(total_smoothness0 / total_bitrate0 * 100.0, 2)
            self.log_file.write(str(bitrate) + ',' +
                                str(rebuffer) + ',' + 
                                str(smoothness))
            self.log_file.write(' ')
        self.log_file.write('\n')
        self.log_file.flush()

    def write_line(self):
        self.log_file.write('\n')
        self.log_file.flush()
