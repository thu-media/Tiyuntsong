import os
import numpy as np
import sabre

ALPHA = 4.3

class tracepool(object):
    def __init__(self, workdir='./traces', ratio=0.1):
        self.work_dir = workdir
        self.trace_list = []
        self.abr_list = [sabre.ThroughputRule, sabre.Bola,
                         sabre.BolaEnh, sabre.DynamicDash, sabre.Dynamic]
        self.sample_list = []
        for p in os.listdir(self.work_dir):
            for l in os.listdir(self.work_dir + '/' + p):
                if np.random.rand() <= ratio:
                    self.trace_list.append(self.work_dir + '/' + p + '/' + l)

        for p in self.abr_list:
            self.sample_list.append([])
        self.sample()

    def sample(self):
        print('generating samples')
        for _trace in self.get_list():
            for _index, _abr in enumerate(self.abr_list):
                self.sample_list[_index].append(
                    sabre.execute_model(abr=_abr, trace=_trace))
        print('done')

    def get_list(self):
        return self.trace_list

    def battle(self, agent_result):
        ret = []
        for p in range(len(agent_result[0])):
            ret.append(self._battle_index(agent_result, p))
        return ret

    def _battle_index(self, agent_result, index):
        ret = []
        for _index in range(len(self.abr_list)):
            tmp = [0, 0, 0]
            for _trace_index in range(len(self.get_list())):
                res = self._battle(
                    [agent_result[_trace_index][index], self.sample_list[_index][_trace_index]])
                if res[0] != 0:
                    tmp[np.argmax(res)] += 1
                tmp[-1] += 1
            ret.append(round(tmp[0] * 100.0 / tmp[-1], 2))
        return ret

    def _battle(self, agent_result):
        total_bitrate0, total_rebuffer0, _ = agent_result[0]
        total_bitrate1, total_rebuffer1, _ = agent_result[1]
        if total_rebuffer0 < total_rebuffer1:
            if total_bitrate0 > total_bitrate1:
                return [1, -1]
            elif total_bitrate0 == total_bitrate1:
                return [1, -1]
            else:
                _cof0 = total_rebuffer0 / total_bitrate0
                _cof1 = total_rebuffer1 / total_bitrate1
                if _cof0 > _cof1:
                    return [-1, 1]
                elif _cof0 == _cof1:
                    return [0, 0]
                else:
                    return [1, -1]
        elif total_rebuffer0 == total_rebuffer1:
            if total_bitrate0 > total_bitrate1:
                return [1, -1]
            elif total_bitrate0 == total_bitrate1:
                return [0, 0]
            else:
                return [-1, 1]
        else:
            if total_bitrate0 > total_bitrate1:
                _cof0 = total_rebuffer0 / total_bitrate0
                _cof1 = total_rebuffer1 / total_bitrate1
                if _cof0 > _cof1:
                    return [-1, 1]
                elif _cof0 == _cof1:
                    return [0, 0]
                else:
                    return [1, -1]
            elif total_bitrate0 == total_bitrate1:
                return [-1, 1]
            else:
                return [-1, 1]
