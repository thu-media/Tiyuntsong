import os
import numpy as np
import sabre
from rules import rules, update_elo, update_elo_2
from tqdm import tqdm


class tracepool(object):
    def __init__(self, workdir='./traces', ratio=0.8):
        self.work_dir = workdir
        #self.abr_list = [sabre.ThroughputRule, sabre.DynamicDash,
        #                 sabre.Dynamic, sabre.Bola, sabre.BolaEnh, sabre.ConstrainRule]
        self.abr_list = [sabre.ThroughputRule, sabre.ConstrainRule]
        self.sample_list = []
        self.trace_list_all = []
        for p in os.listdir(self.work_dir):
            for l in os.listdir(self.work_dir + '/' + p):
                if np.random.rand() <= ratio:
                    self.trace_list_all.append(
                        self.work_dir + '/' + p + '/' + l)
        _trace_len = int(len(self.trace_list_all) * ratio)
        self.trace_list = self.trace_list_all[:_trace_len]
        self.test_list = self.trace_list_all[_trace_len+1:]
        self.elo_score = []

        for p in self.abr_list:
            self.sample_list.append([])
            self.elo_score.append(1000.0)
        self.sample()

    def sample(self):
        print('generating samples')
        for _trace in tqdm(self.get_list(),ascii=True):
            for _index, _abr in enumerate(self.abr_list):
                self.sample_list[_index].append(
                    sabre.execute_model(abr=_abr, trace=_trace))

        for _index0 in range(len(self.abr_list)):
            _battle = []
            for _index in range(len(self.abr_list)):
                tmp = [0, 0, 0]
                for _trace_index in range(len(self.get_list())):
                    res = rules([self.sample_list[_index0][_trace_index],
                                 self.sample_list[_index][_trace_index]])
                    if _index0 < _index:
                        self.elo_score = update_elo(self.elo_score,
                                                    _index0, _index, res)
                    tmp[np.argmax(res)] += 1
                    tmp[-1] += 1
                _battle.append(round(tmp[0] * 100.0 / tmp[-1], 2))
            print(_index0, _battle)
        log_file = open('elo_baseline.txt', 'w')
        for p in self.elo_score:
            log_file.write(str(p) + ' ')
        log_file.close()
        print(self.elo_score)

    def get_list(self):
        return self.trace_list

    def get_list_shuffle(self):
        np.random.shuffle(self.trace_list)
        return self.trace_list

    def battle(self, agent_elo, agent_result):
        ret = []
        for p in range(len(agent_result[0])):
            res, agent_elo = self._battle_index(agent_elo, agent_result, p)
            ret.append(res)
        return ret, agent_elo

    def _battle_index(self, agent_elo, agent_result, index):
        ret = []
        for _index in range(len(self.abr_list)):
            tmp = [0, 0, 0]
            for _trace_index in range(len(self.get_list())):
                res = rules(
                    [agent_result[_trace_index][index], self.sample_list[_index][_trace_index]])
                agent_elo = update_elo_2(
                    agent_elo, self.elo_score, index, _index, res)
                # if res[0] != 0:
                tmp[np.argmax(res)] += 1
                tmp[-1] += 1
            ret.append(round(tmp[0] * 100.0 / tmp[-1], 2))
        return ret, agent_elo

    # def _battle(self, agent_result):
    #     total_bitrate0, total_rebuffer0, _ = agent_result[0]
    #     total_bitrate1, total_rebuffer1, _ = agent_result[1]
    #     if total_rebuffer0 < total_rebuffer1:
    #         if total_bitrate0 > total_bitrate1:
    #             return [1, 0]
    #         elif total_bitrate0 == total_bitrate1:
    #             return [1, 0]
    #         else:
    #             _cof0 = total_rebuffer0 / total_bitrate0
    #             _cof1 = total_rebuffer1 / total_bitrate1
    #             if _cof0 > _cof1:
    #                 return [0, 1]
    #             elif _cof0 == _cof1:
    #                 return [1, 0]
    #             else:
    #                 return [1, 0]
    #     elif total_rebuffer0 == total_rebuffer1:
    #         if total_bitrate0 > total_bitrate1:
    #             return [1, 0]
    #         elif total_bitrate0 == total_bitrate1:
    #             return [1, 0]
    #         else:
    #             return [0, 1]
    #     else:
    #         if total_bitrate0 > total_bitrate1:
    #             _cof0 = total_rebuffer0 / total_bitrate0
    #             _cof1 = total_rebuffer1 / total_bitrate1
    #             if _cof0 > _cof1:
    #                 return [0, 1]
    #             elif _cof0 == _cof1:
    #                 return [1, 0]
    #             else:
    #                 return [1, 0]
    #         elif total_bitrate0 == total_bitrate1:
    #             return [0, 1]
    #         else:
    #             return [0, 1]
