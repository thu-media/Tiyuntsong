
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import plot, savefig
from matplotlib import colors
#from pykalman import KalmanFilter
import sys

LW = 4
SCALE = 300


def read_csv(filename='elo.txt'):
    _file = open(filename, 'r')
    _tmp = []
    _tmpA, _tmpB = [], []
    for _line in _file:
        _lines = _line.split(' ')
        if len(_tmp == 0):
            pass
        _tmpA.append(float(_lines[0]))
        _tmpB.append(float(_lines[1]))
    _file.close()
    return [np.array(_tmpA), np.array(_tmpB)]


def read_history(filename='elo_baseline.txt'):
    _file = open(filename, 'r')
    _tmpA = []
    for _line in _file:
        for _l in _line.split(' '):
            if _l is not '':
                _tmpA.append(float(_l))
    _file.close()
    return np.array(_tmpA)


def moving_average(data, alpha=0.9):
    _tmp = []
    _val = data[0]
    for p in data:
        _val = p * (1.0 - alpha) + _val * alpha
        _tmp.append(_val)
    return np.array(_tmp)

def color():
    color_map = []
    for _, hex in matplotlib.colors.cnames.iteritems():
        color_map.append(hex)
    return color_map

plt.switch_backend('Agg')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.labelweight'] = 'bold'
font = {'size': 12}
_color_map = color()
matplotlib.rc('font', **font)
fig, ax1 = plt.subplots(figsize=(20, 12), dpi=50)
_agent = read_csv()
_tmp = read_history()
ax1.grid(True)
ax1.set_title('elo')
_label = ['ThroughputRule', 'DynamicDash', 'Dynamic', 'Bola', 'BolaEnh']
_color = ['darkred', 'darkblue', 'salmon', 'gray', 'pink']
_agent_color = ['darkred', 'darkblue', 'salmon', 'gray', 'pink']
for index, p in enumerate(_agent):
    l4 = ax1.plot(p, color=_color_map[index], lw=LW, alpha=0.2)
    l4 = ax1.plot(moving_average(p), color=_color_map[index], lw=LW, label=str(index))

for index, p in enumerate(_tmp):
    ax1.hlines(p, 0, len(_agent[0]), linestyles="dashed",
               color=_color[index], label=_label[index], lw=LW)
ax1.legend()
savefig('elo.png')
print 'done'
