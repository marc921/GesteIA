import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import norm, get_velocity_norm, get_acc, get_acc_2, convolve_xy, dt, get_gaussian_kernel, get_velocity

# Parameters
width = 480
height = 270
openpose_files_path = os.path.join('.', 'OpenposeFiles')


# Process Openpose output files
person_1 = pd.read_csv('video_coordinates_1.csv')
person_2 = pd.read_csv('video_coordinates_2.csv')

# Position plotting
t = [dt*i for i in range(len(person_1["RWrist_x"]))]

gaussian_kernel = get_gaussian_kernel(7)
x_rh1, y_rh1 = convolve_xy(person_1["RWrist_x"].values, person_1["RWrist_y"].values, gaussian_kernel)
x_lh1, y_lh1 = convolve_xy(person_1["LWrist_x"].values, person_1["LWrist_y"].values, gaussian_kernel)
x_rh2, y_rh2 = convolve_xy(person_2["RWrist_x"].values, person_2["RWrist_y"].values, gaussian_kernel)
x_lh2, y_lh2 = convolve_xy(person_2["LWrist_x"].values, person_2["LWrist_y"].values, gaussian_kernel)

x_rh = [x_rh1, x_rh2]
y_rh = [y_rh1, y_rh2]
x_lh = [x_lh1, x_lh2]
y_lh = [y_lh1, y_lh2]

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(t, y_rh1, 'r-', label='Right hand 1')
ax[0].plot(t, y_lh1, 'b-', label='Left hand 1')
ax[0].legend(loc='upper left')

ax[1].plot(t, y_rh2, 'r-', label='Right hand 2')
ax[1].plot(t, y_lh2, 'b-', label='Left hand 2')
ax[1].legend(loc='upper left')

plt.show()

# Velocity computing and plotting
v_r1 = get_velocity_norm(x_rh1, y_rh1)
v_l1 = get_velocity_norm(x_lh1, y_lh1)

v_r1_x, v_r1_y = get_velocity(x_rh1, y_rh1)
v_l1_x, v_l1_y = get_velocity(x_lh1, y_lh1)

t_v = [dt*i for i in range(len(v_r1))]

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(t_v, v_r1_y, 'r-', label='Right hand velocity 1')
ax[0].plot(t_v, v_l1_y, 'g-', label='Left hand velocity 1')
ax[0].legend(loc='upper left')

ax[1].plot(t_v, v_r1_y, 'r-', label='Right hand velocity 1')
ax[1].plot(t_v, v_l1_y, 'g-', label='Left hand velocity 1')
ax[1].legend(loc='upper left')
plt.show()

# Acceleration computing and plotting
a_r1 = get_acc(x_rh1, y_rh1)
a_l1 = get_acc(x_lh1, y_lh1)
a_r2 = get_acc(x_rh2, y_rh2)
a_l2 = get_acc(x_lh2, y_lh2)

t_a = [dt*i for i in range(len(a_r1))]

# Trying something
a_r1_x, a_r1_y = get_velocity(v_r1_x, v_r1_y)
a_l1_x, a_l1_y = get_velocity(v_l1_y, v_l1_y)

plt.plot(t_a, a_r1_y, 'r-', label='Right hand acceleration 1')
plt.plot(t_a, a_l1_y, 'g-', label='Left hand acceleration 1')
plt.legend(loc='upper left')
plt.show()
