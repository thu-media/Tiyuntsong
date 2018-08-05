import sabre as env
import math
from network import Zero
from tracepool import tracepool
import numpy as np
from tqdm import tqdm
log_file = open('zero.txt', 'w')


def battle(agent_result):
    global log_file
    total_bitrate0, total_rebuffer0, _ = agent_result[0]
    total_bitrate1, total_rebuffer1, _ = agent_result[1]
    log_file.write(str(total_bitrate0) + ',' + str(round(total_rebuffer0)) +
                   ',' + str(total_bitrate1) + ',' + str(round(total_rebuffer1)))
    log_file.write('\n')
    log_file.flush()
    if total_rebuffer0 < total_rebuffer1:
        return [1, -1]
    elif total_rebuffer0 == total_rebuffer1:
        if total_bitrate0 > total_bitrate1:
            return [1, -1]
        elif total_bitrate0 == total_bitrate1:
            return [0, 0]
        else:
            return [-1, 1]
    else:
        return [-1, 1]


def main():
    global log_file
    agent_list = [Zero('A'), Zero('B')]
    _update, _clear = 0, 1
    _tracepool = tracepool(ratio=0.01)
    while True:
        _tmp = [0, 0, 0]
        _state_stack, _reward_stack = [], []
        for _trace in tqdm(_tracepool.get_list()):
            agent_result = []
            for _agent in agent_list:
                total_bitrate, total_rebuffer, total_smoothness = env.execute(
                    abr=_agent, trace=_trace)
                agent_result.append(
                    (total_bitrate, total_rebuffer, total_smoothness))
            agent_reward = battle(agent_result)
            #print(agent_reward, _tmp)
            _tmp[np.argmax(agent_reward)] += 1
            _tmp[-1] += 1
            agent_list[_update].update(agent_reward[_update])
            agent_list[_clear].clear()
            # for _agent, _r in zip(agent_list, agent_reward):
            #    _agent.update(_r)
        _clear = np.argmax(_tmp[0:-1])
        _update = np.argmin(_tmp[0:-1])
        print(round(_tmp[0] * 100.0 / _tmp[-1], 2), '%')
        log_file.write('\n')
        log_file.flush()


if __name__ == '__main__':
    main()
