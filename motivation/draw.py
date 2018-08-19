
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


def read_log(filename):
    _file = open(filename, 'r')
    _bitrate, _throughput = [], []
    for _line in _file:
        _lines = _line.split()
        if len(_lines) > 0:
            _bitrate.append(float(_lines[1]))
            _throughput.append(float(_lines[4]) / float(_lines[5]) * 8.0)
    _file.close()
    return np.array(_bitrate), np.array(_throughput)


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

plt.rcParams['axes.labelsize'] = 50
plt.rcParams['axes.labelweight'] = 'bold'
font = {'size': 50}
_color_map = color()
matplotlib.rc('font', **font)
os.system('mkdir imgs')
for p in os.listdir('./test_results'):
    _filename = './test_results/' + p
    print _filename
    fig, ax1 = plt.subplots(figsize=(30, 15), dpi=50)
    ax1.grid(True)
    ax1.set_title(r'\textbf{' + p.replace('_', '-') + r'}')
    ax1.set_xlabel(r'\textbf{time$(s)$}')
    ax1.set_ylabel(r'\textbf{video bitrate$(Kbps)$}')
    _bitrate, _throughput = read_log(_filename)
    l4 = ax1.plot(_bitrate, '--', color='darkred', label=r'bitrate', lw=LW)
    l4 = ax1.plot(_throughput, color='darkblue', label=r'throughput', lw=LW)
    ax1.legend(fontsize=50)
    savefig('./imgs/' + p + '.pdf')
print 'done'
