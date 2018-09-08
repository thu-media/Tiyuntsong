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
AUTO_SAVE = True
TEST_EPOCH = 5
SAMPLE_LEN = 30
MODEL_SAVE_NAME = 'model/best'
os.system('mkdir model')
def main():
    _max_score = 0.0
    _log = log('zero.txt')
    #log_file = open('zero.txt', 'w')
    elo_file = open('elo.txt', 'w')
    agent_list = []
    agent_elo = []
    _tracepool = tracepool()
    for p in range(NUM_AGENT):
        agent_list.append(Zero(str(p)))
        agent_elo.append(1000.0)
    _epoch = 0
    while True:
        #for _agent in agent_list:
        #    _agent.set_test(False)

        pbar = tqdm(total=TEST_EPOCH * SAMPLE_LEN, ascii=True)
        for p in range(TEST_EPOCH):
            _tmp = [0, 0, 0]
            _state_stack, _reward_stack = [], []
            _trace_result = []
            for _trace in _tracepool.get_list_shuffle(sample=SAMPLE_LEN):
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
                pbar.update(1)
            _delta_array = [_tmp[0] / _tmp[-1], _tmp[1] / _tmp[-1]]

            for _agent, _d in zip(agent_list, _delta_array):
                _agent.learn(_d)
                _agent.clear()
            _epoch += 1
        pbar.close()
        #start test
        #for _agent in agent_list:
        #    _agent.set_test(True)
        agent_elo = []
        for p in range(NUM_AGENT):
            agent_elo.append(1000.0)
        _trace_result = []
        print(_epoch, 'start testing...')
        for _trace in tqdm(_tracepool.get_test_set(),ascii=True):
            agent_result = []
            for _agent in agent_list:
                total_bitrate, total_rebuffer, total_smoothness = env.execute(
                    abr=_agent, trace=_trace)
                agent_result.append(
                    (total_bitrate, total_rebuffer, total_smoothness))
                _trace_result.append(agent_result)
                _agent.clear()
        _rate, agent_elo = _tracepool.battle(agent_elo, _trace_result)
        for p in agent_elo:
            elo_file.write(str(p) + ' ')
        elo_file.write('\n')
        elo_file.flush()
        print(_rate)
        print(agent_elo)
        _elo_score = np.max(agent_elo)
        if _elo_score > _max_score:
            _max_score = _elo_score
            if AUTO_SAVE:
                agent_list[np.argmax(agent_elo)].save(MODEL_SAVE_NAME)
                model_log = open(MODEL_SAVE_NAME + ".txt", 'a')
                model_log.write(str(_max_score) + '\n')
                model_log.close()
                print('Auto saved')

        for _agent in agent_list:
            _agent.save_current()

        print(round(_tmp[0] * 100.0 / _tmp[-1], 2), '%',
            ',', round(_tmp[1] * 100.0 / _tmp[-1], 2), '%')
            
        _log.write_line()
        os.system('python3 draw.py')


if __name__ == '__main__':
    main()
