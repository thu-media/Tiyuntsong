import sabre as env
import math
from network import Zero
from tracepool import tracepool
import numpy as np
from tqdm import tqdm
log_file = open('zero.txt', 'w')
NUM_AGENT = 10


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
    agent_list = []
    for p in range(NUM_AGENT):
        agent_list.append(Zero(scope=str(p)))
    _update, _clear = 0, 1
    _tracepool = tracepool(ratio=0.01)
    while True:
        _tmp = np.zeros((NUM_AGENT + 1))
        _state_stack, _reward_stack = [], []
        _trace_result = []
        for _trace in tqdm(_tracepool.get_list()):
            agent_result = []
            for _agent in agent_list:
                total_bitrate, total_rebuffer, total_smoothness = env.execute(
                    abr=_agent, trace=_trace)
                agent_result.append(
                    (total_bitrate, total_rebuffer, total_smoothness))
            _index = [0, 0]
            #_agent_map = {}
            for _index_0, _agent_0 in enumerate(agent_result):
                _index[0] = _index_0
                #_agent_map[_index_0] = []
                _simple_battle = np.zeros((3))
                for _index_1, _agent_1 in enumerate(agent_result):
                    _index[1] = _index_1
                    agent_reward = battle([_agent_0, _agent_1])
                    _simple_battle[np.argmax(agent_reward)] += 1
                    _simple_battle[-1] += 1
                    #_agent_map[_index_0].append(agent_reward)
                _tmp[_index[np.argmax(_simple_battle[0:-1])]] += 1
                _tmp[-1] += 1
            # for _agent, _r in zip(agent_list, agent_reward):
            #    _agent.push(_r)
            _trace_result.append(agent_result)
        _clear = np.argmax(_tmp[0:-1])
        for p in range(len(agent_list)):
            for t in _tracepool.get_list():
                agent_reward = battle([
                    _trace_result[t][p], _trace_result[t][_clear]])
                agent_list[p]
        #_buffer = agent_list[_clear].pull()
        for p in range(len(agent_list)):
            if p != _clear:
                agent_list[p].learn()
                # agent_list[p].learn()
        for _agent in agent_list:
            _agent.clear()
        print(_tracepool.battle(_trace_result))
        print(round(_tmp[_clear] * 100.0 / _tmp[-1], 2), '%')
        log_file.write('\n')
        log_file.flush()


if __name__ == '__main__':
    main()
