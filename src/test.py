import math


def ret():
    return [1, 2, 3, 4, 5, 6]


def chunks(arr, m):
    if (len(arr) < m):
        m = len(arr)
    tmp = []
    idx = 0
    for i in range(m):
        tmp.append([])
    for i in range(len(arr)):
        tmp[idx].append(arr[i])
        idx += 1
        idx %= m
    return tmp


m = ret()
print chunks(m, 6)
