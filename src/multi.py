import sabre as env
import math
from network import Zero
from tracepool import tracepool
import numpy as np
from tqdm import tqdm
from rules import rules
from log import log
import os
from multiprocessing import cpu_count
import multiprocessing as mp

NUM_AGENT = 2
USE_CORES = cpu_count()


def agent(agent_id, net_params_queue, exp_queue):
    agent_list = []
    for p in range(NUM_AGENT):
        agent_list.append(Zero(str(p)))

    while True:
        net_params, _tracepool = net_params_queue.get()
        for i in range(NUM_AGENT):
            agent_list[i].set_params(net_params[i])

        _trace_result = []
        _global_history = []
        for p in range(NUM_AGENT):
            _global_history.append([])
        for _trace in _tracepool:
            agent_result = []
            for _agent in agent_list:
                total_bitrate, total_rebuffer, total_smoothness = env.execute(
                    abr=_agent, trace=_trace)
                agent_result.append(
                    (total_bitrate, total_rebuffer, total_smoothness))
            agent_reward = []
            for _index in range(len(agent_list[0].quality_history)):
                res = rules([agent_list[0].quality_history[_index],
                             agent_list[1].quality_history[_index]])
                agent_reward.append(res)
            agent_reward = np.array(agent_reward)
            for _index, _agent in enumerate(agent_list):
                _history = _agent.get_action()
                reward = agent_reward[:, _index]
                _idx = 0
                s_batch, a_batch, r_batch, g_batch = [], [], [], []
                for (state, action, gan) in _history:
                    s_batch.append(state)
                    a_batch.append(action)
                    r_batch.append(reward[_idx])
                    g_batch.append(gan)
                    _idx += 1
                _global_history[_index].append(
                    (s_batch, a_batch, r_batch, g_batch))
            _trace_result.append(agent_result)
        exp_queue.put([_global_history, _trace_result])


def chunks(arr, m):
    if (len(arr) < m):
        m = len(arr)
    tmp, tmp_index = [], []
    idx = 0
    for i in range(m):
        tmp.append([])
        tmp_index.append([])
    for i in range(len(arr)):
        tmp[idx].append(arr[i])
        tmp_index[idx].append(i)
        idx += 1
        idx %= m
    return tmp, tmp_index


def central(net_params_queues, exp_queues):
    global_agent_list = []
    agent_elo = []

    _log = log('zero.txt')
    #log_file = open('zero.txt', 'w')
    elo_file = open('elo.txt', 'w')
    for p in range(NUM_AGENT):
        global_agent_list.append(Zero(str(p)))
        agent_elo.append(1000.0)

    _tracepool = tracepool(ratio=0.1)
    _split_pool, _idx_pool = chunks(_tracepool.get_list(), USE_CORES)
    while True:
        # synchronize the network parameters of work agent
        _params = []
        agent_elo = []
        global_trace_pool = []
        for p in range(len(_tracepool.get_list())):
            global_trace_pool.append([])

        for p in range(NUM_AGENT):
            agent_elo.append(1000.0)
            _params.append(global_agent_list[p].get_params())

        for i in range(USE_CORES):
            net_params_queues[i].put([_params, _split_pool[i]])

        _tmp = [0, 0, 0]
        for i in tqdm(range(USE_CORES), ascii=True):
            _global_history, _trace_result = exp_queues[i].get()
            for p in range(NUM_AGENT):
                _history = _global_history[p]
                global_agent_list[p].set_action(_history)
            for p in range(len(_trace_result)):
                global_trace_pool[_idx_pool[i][p]] = _trace_result[p]
        for _trace_res in global_trace_pool:       
            tmp_battle = rules(_trace_res)
            _tmp[np.argmax(tmp_battle)] += 1
            _tmp[-1] += 1
        _rate, agent_elo = _tracepool.battle(agent_elo, global_trace_pool)
        _delta_array = [_tmp[0] / _tmp[-1], _tmp[1] / _tmp[-1]]
        for _agent, _d in zip(global_agent_list, _delta_array):
            _agent.learn(_d)
            _agent.clear()

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


def main():
    net_params_queues = []
    exp_queues = []
    for i in range(USE_CORES):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    coordinator = mp.Process(target=central,
                             args=(net_params_queues, exp_queues))
    coordinator.start()
    agents = []
    for i in range(USE_CORES):
        agents.append(mp.Process(target=agent,
                                 args=(i, net_params_queues[i], exp_queues[i])))
    for p in agents:
        p.start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
