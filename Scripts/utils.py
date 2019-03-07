import math
import numpy as np

# Video settings
dt = 1/25


def norm(x, y, z=0):
    return math.sqrt(x**2 + y**2 + z**2)


def gauss(x):
    return math.exp(- x**2) / math.sqrt(2 * math.pi)


def get_gaussian_kernel(size=3):
    l = np.linspace(-1, 1, num=size)
    res = np.array([gauss(e) for e in l])
    return res / np.linalg.norm(res)


def get_velocity(x, y):
    vx, vy = [], []
    for i in range(len(x) - 1):
        vx += [(x[i+1] - x[i]) / dt]
        vy += [(y[i+1] - y[i]) / dt]
    return vx, vy


def get_velocity_3D(x, y, z):
    vx, vy = [], []
    for i in range(len(x) - 1):
        vx += [(x[i+1] - x[i]) / dt]
        vy += [(y[i+1] - y[i]) / dt]
        vz += [(z[i+1] - z[i]) / dt]
    return vx, vy, vz


def get_velocity_norm(x, y):
    v = []
    for i in range(len(x) - 1):
        vx = (x[i+1] - x[i]) / dt
        vy = (y[i+1] - y[i]) / dt
        v += [norm(vx, vy)]
    return v


def get_velocity_norm_3D(x, y, z):
    v = []
    for i in range(len(x) - 1):
        vx = (x[i+1] - x[i]) / dt
        vy = (y[i+1] - y[i]) / dt
        vz = (z[i+1] - z[i]) / dt
        v += [norm(vx, vy)]
    return v


def get_acc(x, y):
    v = get_velocity_norm(x, y)
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


def moyenne_glissante(x, n):
    x_c = []
    for i in range(len(x) - n):
        x_c += [np.array(x[i:i+n]).mean()]
    return np.array(x_c)


def convolve_xyz(x, y, z, kernel):
    x_c = np.convolve(x, kernel, mode='same')
    y_c = np.convolve(y, kernel, mode='same')
    z_c = np.convolve(z, kernel, mode='same')
    return x_c, y_c, z_c


def remove_max_outliers(x, ratio=0.05):
    x_inliers = np.array(x)
    quantile = np.quantile(x_inliers, max(0, 1-ratio), interpolation='lower')
    x_inliers[x_inliers >= quantile] = 0
    return x_inliers


def get_acc_from_v(v):
    acc = []
    for i in range(len(v) - 1):
        acc += [(v[i+1] - v[i]) / dt]
    return np.array(acc)
