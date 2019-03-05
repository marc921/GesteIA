import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import get_velocity_norm_3D, get_velocity_norm, convolve_xy, get_gaussian_kernel, dt, remove_max_outliers, convolve_xyz, moyenne_glissante

# Settings
data_path = os.path.join('..', 'Data')
mocap_label_x = 'LeftHand_x'
mocap_label_y = 'LeftHand_y'
mocap_label_z = 'LeftHand_z'

openpose_label_x = 'LWrist_x'
openpose_label_y = 'LWrist_y'

# Mocap files
mocap_df1 = pd.read_csv(os.path.join(data_path, 'Mocap_1.csv'), sep=';')
mocap_df2 = pd.read_csv(os.path.join(data_path, 'Mocap_2.csv'), sep=';')

# Keeping only position coordinates
mocap_df1 = mocap_df1[mocap_df1['Type'] == 'position']
mocap_df2 = mocap_df2[mocap_df2['Type'] == 'position']

# Creation of variables per coordinates
xm1, ym1, zm1 = mocap_df1[mocap_label_x].values, mocap_df1[mocap_label_y].values, mocap_df1[mocap_label_z].values
xm2, ym2, zm2 = mocap_df2[mocap_label_x].values, mocap_df2[mocap_label_y].values, mocap_df2[mocap_label_z].values

# Process Openpose output files
person_1 = pd.read_csv(os.path.join(data_path, 'video_coordinates_1.csv'))
person_2 = pd.read_csv(os.path.join(data_path, 'video_coordinates_2.csv'))

# Position
t = [dt*i for i in range(len(person_1["RWrist_x"]))]

n = 500
x_lh1 = person_1[openpose_label_x].values
y_lh1 = person_1[openpose_label_y].values
x_lh2 = person_2[openpose_label_x].values
y_lh2 = person_2[openpose_label_y].values

# Velocity
# Openpose
v_l1 = get_velocity_norm(x_lh1, y_lh1)
v_l2 = get_velocity_norm(x_lh2, y_lh2)

# Mocap
vm_l1 = get_velocity_norm_3D(xm1, ym1, zm1)
vm_l2 = get_velocity_norm_3D(xm2, ym2, zm2)

# Moyenne glissante
v_l1 = moyenne_glissante(v_l1, n)
v_l2 = moyenne_glissante(v_l2, n)
vm_l1 = moyenne_glissante(vm_l1, n)
vm_l2 = moyenne_glissante(vm_l2, n)

t_v = [dt*i for i in range(len(v_l1))]
t_vm1 = mocap_df1['Time'].values[:len(vm_l1)]
t_vm2 = mocap_df2['Time'].values[:len(vm_l2)] - 5

# removing some outliers
# v_l1 = remove_max_outliers(v_l1, 0.01)
# v_l2 = remove_max_outliers(v_l2, 0.01)
# vm_l1 = remove_max_outliers(vm_l1, 0.01)
# vm_l2 = remove_max_outliers(vm_l2, 0.01)

fig, ax = plt.subplots(nrows=2, ncols=1)
# ax[0][0].plot(t_v, v_r1_y, 'r-', label='Right hand velocity 1')
ax[0].plot(t_v, v_l1, 'g-', label='Left hand velocity 1')
ax[0].plot(t_vm1, vm_l1 * 4000, 'r-', label='Left hand velocity 1 - MOCAP')
ax[0].legend(loc='upper left')

# ax[0][1].plot(t_v, v_r2_y, 'r-', label='Right hand velocity 2')
ax[1].plot(t_v, v_l2, 'g-', label='Left hand velocity 2')
ax[1].plot(t_vm2, vm_l2 * 4000, 'r-', label='Left hand velocity 2 - MOCAP')
ax[1].legend(loc='upper left')

# To maximize the plots
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')

plt.show()
