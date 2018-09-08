
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import plot, savefig
#from pykalman import KalmanFilter
import sys

LW = 4
SCALE = 300


def read_csv(filename='elo.txt'):
    _file = open(filename, 'r')
    _tmpA, _tmpB = [], []
    for _line in _file:
        _lines = _line.split(' ')
        _tmpA.append(float(_lines[0]))
        _tmpB.append(float(_lines[1]))
    _file.close()
    return np.array(_tmpA), np.array(_tmpB)


def read_history(filename='elo_baseline.txt'):
    _file = open(filename, 'r')
    _tmpA = []
    for _line in _file:
        for _l in _line.split(' '):
            if _l is not '':
                _tmpA.append(float(_l))
    _file.close()
    return np.array(_tmpA)


def moving_average(data, alpha=0.60):
    _tmp = []
    _val = data[0]
    for p in data:
        _val = p * (1.0 - alpha) + _val * alpha
        _tmp.append(_val)
    return np.array(_tmp)

plt.switch_backend('Agg')

plt.rcParams['axes.labelsize'] = 40
plt.rcParams['axes.labelweight'] = 'bold'
font = {'size': 30}
matplotlib.rc('font', **font)
fig, ax1 = plt.subplots(figsize=(20, 12), dpi=50)
_a, _b = read_csv()
_tmp = read_history()
ax1.grid(True)
ax1.set_title('Tiyuntsong')
ax1.set_ylabel('elo')
ax1.set_xlabel('step')
l4 = ax1.plot(_a, color='darkgreen', lw=LW, alpha=0.2)
l4 = ax1.plot(moving_average(_a), color='darkgreen', lw=LW, label='A')
l4 = ax1.plot(_b, color='darkblue', lw=LW, alpha=0.2)
l4 = ax1.plot(moving_average(_b), color='darkblue', lw=LW, label='B')
_label = ['ThroughputRule', 'DynamicDash', 'Dynamic', 'Bola', 'BolaEnh', 'Constrained']
#_label = ['ThroughputRule', 'DynamicDash', 'Dynamic', 'Bola', 'BolaEnh', 'Constrained']
_color = ['darkred', 'darkblue', 'salmon', 'gray', 'pink', 'darkgreen']
for index, p in enumerate(_tmp):
    ax1.hlines(p, 0, len(_a), linestyles="dashed", color = _color[index], lw = LW)
ax1.legend()
savefig('elo.png')
os.system('scp elo.png root@1029.mythkast.net:/root/elo/7.png')
