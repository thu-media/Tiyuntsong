import sabre as env
import math
from network import Zero

m_a = 0
m_b = 0
m_c = 0
def battle(agent_result):
    global m_a, m_b, m_c
    if m_b != 0:
        print(round(m_a * 100.0 / m_c, 2), round(m_b * 100.0 / m_c, 2), round((m_c - m_a - m_b) * 100.0 / m_c, 2))
    total_bitrate0, total_rebuffer0, _ = agent_result[0]
    total_bitrate1, total_rebuffer1, _ = agent_result[1]
    #print(total_bitrate0,total_bitrate1,total_rebuffer0,total_rebuffer1)
    m_c += 1
    if total_rebuffer0 == total_rebuffer1 and total_bitrate0 == total_bitrate1:
        return [0,0]
    if total_rebuffer0 <= total_rebuffer1:
        if total_bitrate0 >= total_bitrate1:
            m_a += 1
            return [1, -1]
        else:
            return [0, 0]
    else:
        if total_bitrate0 < total_bitrate1:
            m_b += 1
            return [-1, 1]
        else:
            return [0, 0]


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
