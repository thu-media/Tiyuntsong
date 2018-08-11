import sabre as env
import math
from network import Zero
from tracepool import tracepool
import numpy as np
from tqdm import tqdm
import argparse
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


def parse(argv=None):
    parser = argparse.ArgumentParser(description='Simulate an ABR session.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', metavar='SAVEFILE', default='save.log',
                        help='Save the secific metrics to file.')
    parser.add_argument('-nm', '--network-multiplier', metavar='MULTIPLIER',
                        type=float, default=1,
                        help='Multiply throughput by MULTIPLIER.')
    parser.add_argument('-ml', '--movie-length', metavar='LEN', type=float, default=None,
                        help='Specify the movie length in seconds (use MOVIE length if None).')
    parser.add_argument('-ab', '--abr-basic', action='store_true',
                        help='Set ABR to BASIC (ABR strategy dependant).')
    parser.add_argument('-ao', '--abr-osc', action='store_true',
                        help='Set ABR to minimize oscillations.')
    parser.add_argument('-gp', '--gamma-p', metavar='GAMMAP', type=float, default=5,
                        help='Specify the (gamma p) product in seconds.')
    parser.add_argument('-noibr', '--no-insufficient-buffer-rule', action='store_true',
                        help='Disable Insufficient Buffer Rule.')
    parser.add_argument('-ws', '--window-size', metavar='WINDOW_SIZE',
                        nargs='+', type=int, default=[3],
                        help='Specify sliding window size.')
    parser.add_argument('-hl', '--half-life', metavar='HALF_LIFE',
                        nargs='+', type=float, default=[3, 8],
                        help='Specify EWMA half life.')
    parser.add_argument('-s', '--seek', nargs=2, metavar=('WHEN', 'SEEK'),
                        type=float, default=None,
                        help='Specify when to seek in seconds and where to seek in seconds.')
    choices = ['none', 'left', 'right']
    parser.add_argument('-r', '--replace', metavar='REPLACEMENT',
                        choices=choices, default='none',
                        help='Set replacement strategy (%s).' % ', '.join(choices))
    parser.add_argument('-b', '--max-buffer', metavar='MAXBUFFER', type=float, default=25,
                        help='Specify the maximum buffer size in seconds.')
    parser.add_argument('-noa', '--no-abandon', action='store_true',
                        help='Disable abandonment.')
    parser.add_argument('-rmp', '--rampup-threshold', metavar='THRESHOLD',
                        type=int, default=None,
                        help='Specify at what quality index we are ramped up (None matches network).')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Run in verbose mode.')
    if argv is None:
        args = parser.parse_args(None)
    else:
        args = parser.parse_args(argv.split(' '))
    return args


def main():
    global log_file
    #agent_list = [Zero('A'), Zero('B')]
    agent_list = []
    args = parse()
    buffer_size = args.max_buffer * 1000
    gamma_p = args.gamma_p
    config = {'buffer_size': buffer_size,
              'gp': gamma_p,
              'abr_osc': args.abr_osc,
              'abr_basic': args.abr_basic,
              'no_ibr': args.no_insufficient_buffer_rule}
    for p in range(NUM_AGENT):
        agent_list.append(Zero(config=config, scope=str(p)))
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
            # if tmp_battle[0] != 0:
            _tmp[np.argmax(tmp_battle)] += 1
            _tmp[-1] += 1
            for _index, _agent in enumerate(agent_list):
                _agent.push(agent_reward[:, _index])
            _trace_result.append(agent_result)
        _delta_array = [_tmp[0] / _tmp[-1], _tmp[1] / _tmp[-1]]
        for _agent, _d in zip(agent_list, _delta_array):
            _agent.learn(_d)
            _agent.clear()
        # _clear = np.argmax(_tmp[0:-1])
        # _update = np.argmin(_tmp[0:-1])
        # agent_list[0].teach(agent_list[1].pull())
        # agent_list[1].teach(agent_list[0].pull())
        print(_tracepool.battle(_trace_result))
        print(round(_tmp[0] * 100.0 / _tmp[-1], 2), '%',
              ',', round(_tmp[1] * 100.0 / _tmp[-1], 2), '%')
        log_file.write('\n')
        log_file.flush()


if __name__ == '__main__':
    main()
