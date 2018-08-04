import os
import numpy as np
class tracepool(object):
    def __init__(self,workdir = './traces',ratio = 0.1):
        self.work_dir = workdir
        self.trace_list = []
        for p in os.listdir(self.work_dir):
            for l in os.listdir(self.work_dir + '/' + p):
                if np.random.rand() <= ratio:
                    self.trace_list.append(self.work_dir + '/' + p + '/' + l)

    def get_list(self):
        return self.trace_list
