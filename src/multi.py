import sabre as env
import math
from network import Zero
from tracepool import tracepool
import numpy as np
from tqdm import tqdm
from rules import rules
from log import log
import os
NUM_AGENT = 3


def main():
    _log = log('zero.txt')
    # log_file = open('zero.txt', 'w')
    elo_file = open('elo.txt', 'w')
    agent_list = []
    agent_elo = []
    for p in range(NUM_AGENT):
        agent_list.append(Zero(str(p)))
        agent_elo.append(1000.0)
    _tracepool = tracepool(ratio=0.01)
    while True:
        _tmp = np.zeros((NUM_AGENT + 1))
        _state_stack, _reward_stack = [], []
        _trace_result = []
        for _trace in tqdm(_tracepool.get_list(), ascii=True):
            agent_result = []
            for _agent in agent_list:
                total_bitrate, total_rebuffer, total_smoothness = env.execute(
                    abr=_agent, trace=_trace)
                agent_result.append(
                    (total_bitrate, total_rebuffer, total_smoothness))
            # agent_reward = []
            # for multi-agent
            agent_reward_dict = {}
            for _index0, _agent0 in enumerate(agent_list):
                for _index1, _agent1 in enumerate(agent_list):
                    assert len(agent_list[_index0].quality_history) == len(
                        agent_list[_index1].quality_history)
                    _agent0 = agent_list[_index0]
                    _agent1 = agent_list[_index1]
                    agent_reward_tmp = []
                    for _index in range(len(_agent0.quality_history)):
                        if _index0 == _index1:
                            agent_reward_tmp.append([0, 0])
                        else:
                            res = rules([_agent0.quality_history[_index],
                                         _agent1.quality_history[_index]])
                            agent_reward_tmp.append(res)
                    agent_reward_tmp = np.array(agent_reward_tmp)
                    agent_reward_dict[(_index0, _index1)] = agent_reward_tmp
            # for multi-agent
            _agent_battle_tmp = np.zeros((NUM_AGENT + 1))
            for _index0, _agent0 in enumerate(agent_list):
                for _index1, _agent1 in enumerate(agent_list):
                    if _index0 != _index1:
                        tmp_battle = rules(
                            [agent_result[_index0], agent_result[_index1]])
                        _agent_battle_tmp[np.argmax(tmp_battle)] += 1
                        _agent_battle_tmp[-1] += 1

            _winner = np.argmax(_agent_battle_tmp[:-1])
            _log.write_log(agent_result)
            # agent_reward=np.array(agent_reward)
            for _index, _agent in enumerate(agent_list):
                if _index != _winner:
                #     for _tmp_i, _ in enumerate(agent_list):
                #         if _tmp_i != _index:
                #             _agent.push(
                #                 agent_reward_dict[(_index, _tmp_i)][:, 0])
                # else:
                    _agent.push(agent_reward_dict[(_winner, _index)][:, 1])
            _trace_result.append(agent_result)
            #_log.write_log(agent_result[_winner])
            _tmp[_winner] += 1
            _tmp[-1] += 1

        _delta_array = np.zeros((NUM_AGENT))
        for p in range(NUM_AGENT):
            _delta_array[p] = _tmp[p] / _tmp[-1]
        print(_tmp, _delta_array)
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
        print(agent_elo)
        print(round(_tmp[0] * 100.0 / _tmp[-1], 2), '%',
              ',', round(_tmp[1] * 100.0 / _tmp[-1], 2), '%')

        _log.write_line()
        os.system('python draw.py')


if __name__ == '__main__':
    main()
