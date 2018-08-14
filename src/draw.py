
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import plot, savefig
from pykalman import KalmanFilter
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


<<<<<<< HEAD
def moving_average(data, alpha=0.6):
=======
def moving_average(data, alpha=0.93):
>>>>>>> d73c9cecf9fe92b2f038713c382365fd7c3841d4
    _tmp = []
    _val = data[0]
    for p in data:
        _val = p * (1.0 - alpha) + _val * alpha
        _tmp.append(_val)
    return np.array(_tmp)

<<<<<<< HEAD
plt.switch_backend('Agg')
=======

>>>>>>> d73c9cecf9fe92b2f038713c382365fd7c3841d4
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.labelweight'] = 'bold'
font = {'size': 12}
matplotlib.rc('font', **font)
fig, ax1 = plt.subplots(figsize=(20, 12), dpi=50)
_a, _b = read_csv()
_tmp = read_history()
ax1.grid(True)
ax1.set_title('elo')
l4 = ax1.plot(_a, color='red', lw=LW, alpha=0.3)
l4 = ax1.plot(moving_average(_a), color='red', lw=LW, label='A')
l4 = ax1.plot(_b, color='blue', lw=LW, alpha=0.3)
l4 = ax1.plot(moving_average(_b), color='blue', lw=LW, label='B')
_label = ['ThroughputRule', 'DynamicDash', 'Dynamic', 'Bola', 'BolaEnh']
for index, p in enumerate(_tmp):
    ax1.hlines(p, 0, len(_a), linestyles="dashed", label = _label[index])
ax1.legend()
savefig('elo.png')
print 'done'
