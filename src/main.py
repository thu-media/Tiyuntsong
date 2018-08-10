import sabre as env
import math
from network import Zero
from tracepool import tracepool
import numpy as np
from tqdm import tqdm
log_file = open('zero.txt', 'w')
NUM_AGENT = 2


def battle(agent_result, LOG=False):
    global log_file
    total_bitrate0, total_rebuffer0, _ = agent_result[0]
    total_bitrate1, total_rebuffer1, _ = agent_result[1]
    if LOG:
        log_file.write(str(total_bitrate0) + ',' + str(round(total_rebuffer0)) +
                       ',' + str(total_bitrate1) + ',' + str(round(total_rebuffer1)))
        log_file.write('\n')
        log_file.flush()
    if total_rebuffer0 < total_rebuffer1:
        if total_bitrate0 > total_bitrate1:
            return [1, 0]
        elif total_bitrate0 == total_bitrate1:
            return [1, 0]
        else:
            _cof0 = total_rebuffer0 / total_bitrate0
            _cof1 = total_rebuffer1 / total_bitrate1
            if _cof0 > _cof1:
                return [0, 1]
            elif _cof0 == _cof1:
                return [0, 0]
            else:
                return [1, 0]
    elif total_rebuffer0 == total_rebuffer1:
        if total_bitrate0 > total_bitrate1:
            return [1, 0]
        elif total_bitrate0 == total_bitrate1:
            return [0, 0]
        else:
            return [0, 1]
    else:
        if total_bitrate0 > total_bitrate1:
            _cof0 = total_rebuffer0 / total_bitrate0
            _cof1 = total_rebuffer1 / total_bitrate1
            if _cof0 > _cof1:
                return [0, 1]
            elif _cof0 == _cof1:
                return [0, 0]
            else:
                return [1, 0]
        elif total_bitrate0 == total_bitrate1:
            return [0, 1]
        else:
            return [0, 1]


def main():
    global log_file
    #agent_list = [Zero('A'), Zero('B')]
    agent_list = []
    for p in range(NUM_AGENT):
        agent_list.append(Zero(str(p)))
    _update, _clear = 0, 1
    _tracepool = tracepool(ratio=0.01)
    while True:
        _tmp = [0, 0, 0]
        _state_stack, _reward_stack = [], []
        _trace_result = []
        for _trace in tqdm(_tracepool.get_list()):
            agent_result = []
            for _agent in agent_list:
                total_bitrate, total_rebuffer, total_smoothness = env.execute(
                    abr=_agent, trace=_trace)
                agent_result.append(
                    (total_bitrate, total_rebuffer, total_smoothness))
            agent_reward = []
            for _index in range(len(agent_list[0].quality_history)):
                agent_reward.append(battle(
                    [agent_list[0].quality_history[_index], agent_list[-1].quality_history[_index]]))
            agent_reward = np.array(agent_reward)
            tmp_battle = battle(agent_result, True)
            #if tmp_battle[0] != 0:
            _tmp[np.argmax(battle(agent_result, True))] += 1
            _tmp[-1] += 1
            for _index, _agent in enumerate(agent_list):
                _agent.push(agent_reward[:, _index])
            _trace_result.append(agent_result)
        _clear = np.argmax(_tmp[0:-1])
        _buffer = agent_list[_clear].pull()
        for p in range(len(agent_list)):
            if p != _clear:
                agent_list[p].learn()
            # agent_list[p].teach(_buffer)
            # agent_list[p].learn()
        for _agent in agent_list:
            _agent.clear()
        print(_tracepool.battle(_trace_result))
        print(round(_tmp[0] * 100.0 / _tmp[-1], 2), '%',
              ',', round(_tmp[1] * 100.0 / _tmp[-1], 2), '%')
        log_file.write('\n')
        log_file.flush()


if __name__ == '__main__':
    main()
