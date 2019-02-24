import math
import numpy as np

# Video settings
dt = 1/25


def norm(x, y):
    return math.sqrt(x**2 + y**2)


def get_velocity(x, y):
    v = []
    for i in range(len(x) - 1):
        vx = (x[i+1] - x[i]) / dt
        vy = (y[i+1] - y[i]) / dt
        v += [norm(vx, vy)]
    return v


def get_acc(v):
    acc = []
    for i in range(len(v) - 1):
        acc += [(v[i+1] - v[i]) / dt]
    return acc


def get_acc_2(x, y):
    acc = []
    for i in range(len(x) - 2):
        ax = 2*(x[i+2] - 2 * x[i+1] + x[i]) / (dt**2)
        ay = 2*(y[i+2] - 2 * y[i+1] + y[i]) / (dt**2)
        acc += [norm(ax, ay)]
    return acc


def convolve_xy(x, y, kernel):
    x_c = np.convolve(x, kernel, mode='same')
    y_c = np.convolve(y, kernel, mode='same')
    return x_c, y_c
