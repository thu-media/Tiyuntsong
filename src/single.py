import sabre as env
import math
from network import Zero

m_a = 0
m_b = 0
m_c = 0
log_file = open('zero.txt','w')

def battle(agent_result):
    global m_a, m_b, m_c, log_file
    total_bitrate0, total_rebuffer0, _ = agent_result[0]
    total_bitrate1, total_rebuffer1, _ = agent_result[1]
    if m_c > 0:
        print(round(m_a * 100.0 / m_c, 2), round(m_b * 100.0 / m_c, 2))
        print(total_bitrate0,total_rebuffer0,total_bitrate1,total_rebuffer1)
        log_file.write(str(round(m_a * 100.0 / m_c, 2)) + ',' + str(total_bitrate0) + ',' + str(total_rebuffer0) + ',' + str(total_bitrate1) + ',' + str(total_rebuffer1))
        log_file.write('\n')
        log_file.flush()
#    total_bitrate0, total_rebuffer0, _ = agent_result[0]
#    total_bitrate1, total_rebuffer1, _ = agent_result[1]
    #print(total_bitrate0,total_rebuffer0,total_bitrate1,total_rebuffer1)
    m_c += 1
    if total_rebuffer0 < total_rebuffer1:
        m_a += 1
        return [1, 0]
    elif total_rebuffer0 == total_rebuffer1:
        if total_bitrate0 > total_bitrate1:
            m_a += 1
            return [1, 0]
        elif total_bitrate0 == total_bitrate1:
            return [0, 0]
        else:
            m_b += 1
            return [0, 1]
    else:
        m_b += 1
        return [0, 1]



def main():
    agent_list = [Zero('A'), Zero('B')]
    while True:
        agent_result = []
        for _agent in agent_list:
            total_bitrate, total_rebuffer, total_smoothness = env.execute(
                abr=_agent, argv='-nm 0.1')
            agent_result.append((total_bitrate, total_rebuffer, total_smoothness))
        agent_reward = battle(agent_result)

        for _agent, _r in zip(agent_list, agent_reward):
            _agent.update(_r)


if __name__ == '__main__':
    main()
