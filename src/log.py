import os
import sys

class log:
    def __init__(self, filename):
        self.filename = filename
        self.log_file = open(self.filename, 'w')
    
    def write_log(self, agent_result):
        total_bitrate0, total_rebuffer0, _ = agent_result[0]
        total_bitrate1, total_rebuffer1, _ = agent_result[1]
        self.log_file.write(str(total_bitrate0) + ',' + str(round(total_rebuffer0)) +
                        ',' + str(total_bitrate1) + ',' + str(round(total_rebuffer1)))
        self.log_file.write('\n')
        self.log_file.flush()
    
    def write_line(self):
        self.log_file.write('\n')
        self.log_file.flush()
