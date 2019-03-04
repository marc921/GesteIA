import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import get_velocity_norm_3D, get_velocity_norm, convolve_xy, get_gaussian_kernel, dt

# Settings
mocap_path = os.path.join('..', 'Data')

# Mocap files
mocap_df1 = pd.read_csv(os.path.join(mocap_path, 'Mocap_1.csv'), sep=';')
mocap_df2 = pd.read_csv(os.path.join(mocap_path, 'Mocap_2.csv'), sep=';')

# Keeping only position coordinates
mocap_df1 = mocap_df1[mocap_df1['Type'] == 'position']
mocap_df2 = mocap_df2[mocap_df2['Type'] == 'position']

# Creation of variables per coordinates
xm1, ym1, zm1 = mocap_df1['LeftHand_x'].values, mocap_df1['LeftHand_y'].values, mocap_df1['LeftHand_z'].values
xm2, ym2, zm2 = mocap_df2['LeftHand_x'].values, mocap_df2['LeftHand_y'].values, mocap_df2['LeftHand_z'].values

# Process Openpose output files
person_1 = pd.read_csv('video_coordinates_1.csv')
person_2 = pd.read_csv('video_coordinates_2.csv')

# Position
t = [dt*i for i in range(len(person_1["RWrist_x"]))]

gaussian_kernel = get_gaussian_kernel(7)
x_rh1, y_rh1 = convolve_xy(person_1["RWrist_x"].values, person_1["RWrist_y"].values, gaussian_kernel)
x_lh1, y_lh1 = convolve_xy(person_1["LWrist_x"].values, person_1["LWrist_y"].values, gaussian_kernel)
x_rh2, y_rh2 = convolve_xy(person_2["RWrist_x"].values, person_2["RWrist_y"].values, gaussian_kernel)
x_lh2, y_lh2 = convolve_xy(person_2["LWrist_x"].values, person_2["LWrist_y"].values, gaussian_kernel)

# Velocity
# Openpose
v_r1 = get_velocity_norm(x_rh1, y_rh1)
v_l1 = get_velocity_norm(x_lh1, y_lh1)
v_r2 = get_velocity_norm(x_rh2, y_rh2)
v_l2 = get_velocity_norm(x_lh2, y_lh2)

# Mocap
vm_l1 = get_velocity_norm_3D(xm1, ym1, zm1)
vm_l2 = get_velocity_norm_3D(xm2, ym2, zm2)

t_v = [dt*i for i in range(len(v_l1))]
t_vm1 = mocap_df1['Time'].values[:len(vm_l1)]
t_vm2 = mocap_df2['Time'].values[:len(vm_l2)]

fig, ax = plt.subplots(nrows=2, ncols=2)
# ax[0][0].plot(t_v, v_r1_y, 'r-', label='Right hand velocity 1')
ax[0][0].plot(t_v, v_l1, 'g-', label='Left hand velocity 1')
ax[0][0].legend(loc='upper left')

# ax[0][1].plot(t_v, v_r2_y, 'r-', label='Right hand velocity 2')
ax[0][1].plot(t_v, v_l2, 'g-', label='Left hand velocity 2')
ax[0][1].legend(loc='upper left')

ax[1][0].plot(t_vm1, vm_l1, 'g-', label='Left hand velocity 1')
ax[1][0].legend(loc='upper left')

ax[1][1].plot(t_vm2, vm_l2, 'g-', label='Left hand velocity 2')
ax[1][1].legend(loc='upper left')

plt.show()
