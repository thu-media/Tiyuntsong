import numpy as np
import elo


def rules(agent_results, threshold=0.01 * 198 * 3000):
    b_0, r_0, _ = agent_results[0]
    b_1, r_1, _ = agent_results[1]
    _tmp = [0, 0]
    if r_0 <= threshold:
        if r_1 <= threshold:
            _win = np.argmax([b_0, b_1])
        elif r_1 > threshold:
            _win = 0
    elif r_0 > threshold:
        if r_1 <= threshold:
            _win = 1
            #_win = np.argmax([b_0, b_1])
        elif r_1 > threshold:
            _win = np.argmin([r_0 / b_0, r_1 / b_1])
    _tmp[_win] = 1.0
    return _tmp


def update_elo(elo_list, i0, i1, res):
    if res[0] > 0:
        elo_list[i0], elo_list[i1] = elo.rate_1vs1(elo_list[i0], elo_list[i1])
    else:
        elo_list[i1], elo_list[i0] = elo.rate_1vs1(elo_list[i1], elo_list[i0])
    return elo_list


def update_elo_2(agent_list, elo_list, i0, i1, res):
    #print(agent_list, elo_list)
    if res[0] > 0:
        agent_list[i0], _ = elo.rate_1vs1(
            agent_list[i0], elo_list[i1])
    else:
        _, agent_list[i0] = elo.rate_1vs1(
            elo_list[i1], agent_list[i0])
    return agent_list


def basic_rules(agent_result, log_file, LOG=False):
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