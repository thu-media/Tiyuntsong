import sabre as env
import math
from network import Zero
from tracepool import tracepool
import numpy as np
from tqdm import tqdm
from rules import rules
from log import log
import os
NUM_AGENT = 2


def main():
    _log = log('zero.txt')
    #log_file = open('zero.txt', 'w')
    elo_file = open('elo.txt', 'w')
    agent_list = []
    agent_elo = []
    for p in range(NUM_AGENT):
        agent_list.append(Zero(str(p)))
        agent_elo.append(1000.0)
    _tracepool = tracepool(ratio=0.01)
    while True:
        _tmp = [0, 0, 0]
        _state_stack, _reward_stack = [], []
        _trace_result = []
        for _trace in tqdm(_tracepool.get_list(),ascii=True):
            agent_result = []
            for _agent in agent_list:
                total_bitrate, total_rebuffer, total_smoothness = env.execute(
                    abr=_agent, trace=_trace)
                agent_result.append(
                    (total_bitrate, total_rebuffer, total_smoothness))
            agent_reward = []
            for _index in range(len(agent_list[0].quality_history)):
                res = rules([agent_list[0].quality_history[_index],
                             agent_list[-1].quality_history[_index]])
                agent_reward.append(res)
            agent_reward = np.array(agent_reward)
            tmp_battle = rules(agent_result)
            _log.write_log(agent_result)
            _tmp[np.argmax(tmp_battle)] += 1
            _tmp[-1] += 1
            for _index, _agent in enumerate(agent_list):
                _agent.push(agent_reward[:, _index])
            _trace_result.append(agent_result)
        _delta_array = [_tmp[0] / _tmp[-1], _tmp[1] / _tmp[-1]]
        for _agent, _d in zip(agent_list, _delta_array):
            _agent.learn(_d)
        # _clear = np.argmax(_tmp[0:-1])
        # _update = np.argmin(_tmp[0:-1])
        # agent_list[_update].teach(agent_list[_clear].pull())
        for _agent in agent_list:
            _agent.clear()
        
        agent_elo = []
        for p in range(NUM_AGENT):
            agent_elo.append(1000.0)
        _rate, agent_elo = _tracepool.battle(agent_elo, _trace_result)
        for p in agent_elo:
            elo_file.write(str(p) + ' ')
        elo_file.write('\n')
        elo_file.flush()
        print(_rate)
        print(agent_elo)
        print(round(_tmp[0] * 100.0 / _tmp[-1], 2), '%',
              ',', round(_tmp[1] * 100.0 / _tmp[-1], 2), '%')
              
        _log.write_line()
        os.system('python draw.py')


if __name__ == '__main__':
    main()
