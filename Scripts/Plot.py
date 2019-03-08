import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast

from utils import get_velocity_norm_3D, get_velocity_norm, convolve_xy, get_gaussian_kernel, dt, remove_max_outliers, convolve_xyz, moyenne_glissante, get_acc_from_v
from TimeFitting import normalize_array, compare_on_window

# Settings
PLOT_VELOCITIES = False
FIT = True

data_path = os.path.join('..', 'Data')
mocap_label_x = 'LeftHand_x'
mocap_label_y = 'LeftHand_y'
mocap_label_z = 'LeftHand_z'

openpose_label_x = 'LWrist_x'
openpose_label_y = 'LWrist_y'

# Mocap files
print(" # Reading MOCAP files \n")
mocap_df1 = pd.read_csv(os.path.join(data_path, 'Mocap_1.csv'), sep=';')
mocap_df2 = pd.read_csv(os.path.join(data_path, 'Mocap_2.csv'), sep=';')

# Keeping only position coordinates
mocap_df1 = mocap_df1[mocap_df1['Type'] == 'position']
mocap_df2 = mocap_df2[mocap_df2['Type'] == 'position']

# Creation of variables per coordinates
xm1, ym1, zm1 = mocap_df1[mocap_label_x].values, mocap_df1[mocap_label_y].values, mocap_df1[mocap_label_z].values
xm2, ym2, zm2 = mocap_df2[mocap_label_x].values, mocap_df2[mocap_label_y].values, mocap_df2[mocap_label_z].values

# Process Openpose output files
print(" # Reading OPENPOSE files \n")
person_1 = pd.read_csv(os.path.join(data_path, 'video_coordinates_tuples_1.csv'), sep=';')
person_2 = pd.read_csv(os.path.join(data_path, 'video_coordinates_tuples_2.csv'), sep=';')

# Position
t = [dt*i for i in range(len(person_1["RWrist"]))]

r = 500
n = 2*r + 1

person_1['LWrist'] = person_1['LWrist'].apply(ast.literal_eval)
T1 = person_1['LWrist'].values
x_lh1 = [e[0] for e in T1]
y_lh1 = [e[1] for e in T1]

person_2['LWrist'] = person_2['LWrist'].apply(ast.literal_eval)
T2 = person_2['LWrist'].values
x_lh2 = [e[0] for e in T2]
y_lh2 = [e[1] for e in T2]

# Velocity
print(" # Computing velocities \n")
# Openpose
v_l1 = get_velocity_norm(x_lh1, y_lh1)
v_l2 = get_velocity_norm(x_lh2, y_lh2)

# Mocap
vm_l1 = get_velocity_norm_3D(xm1, ym1, zm1)
vm_l2 = get_velocity_norm_3D(xm2, ym2, zm2)

# Moyenne glissante
print(" # Post processing velocities")
v_l1 = moyenne_glissante(v_l1, n)
v_l2 = moyenne_glissante(v_l2, n)
vm_l1 = moyenne_glissante(vm_l1, n)
vm_l2 = moyenne_glissante(vm_l2, n)

t_v = [dt*(i + r) for i in range(len(v_l1))]
t_vm1 = mocap_df1['Time'].values[r:len(vm_l1)+r]
t_vm2 = mocap_df2['Time'].values[r:len(vm_l2)+r]

v_l1 = normalize_array(v_l1)
v_l2 = normalize_array(v_l2)
vm_l1 = normalize_array(vm_l1)
vm_l2 = normalize_array(vm_l2)

if PLOT_VELOCITIES:
    fig, ax = plt.subplots(nrows=2, ncols=1)
    # ax[0][0].plot(t_v, v_r1_y, 'r-', label='Right hand velocity 1')
    ax[0].plot(t_v, v_l1, 'g-', label='Left hand velocity 1')
    ax[0].plot(t_vm1, vm_l1, 'r-', label='Left hand velocity 1 - MOCAP')
    ax[0].legend(loc='upper left')

    # ax[0][1].plot(t_v, v_r2_y, 'r-', label='Right hand velocity 2')
    ax[1].plot(t_v, v_l2, 'g-', label='Left hand velocity 2')
    ax[1].plot(t_vm2, vm_l2, 'r-', label='Left hand velocity 2 - MOCAP')
    ax[1].legend(loc='upper left')

    # To maximize the plots
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()

# trying to fit l1
if FIT:
    print("\n\n\n # Trying to fit graphs \n")

    # To speed up computations
    v_l1_fit = v_l1
    t_v_fit = t_v

    df_openpose = pd.DataFrame(np.zeros(shape=(len(v_l1_fit), 2)), columns=['Time', 'Data'])
    df_openpose['Time'] = t_v_fit
    df_openpose['Data'] = v_l1_fit

    df_mocap = pd.DataFrame(np.zeros(shape=(len(vm_l1), 2)), columns=['Time', 'Data'])
    df_mocap['Time'] = t_vm1
    df_mocap['Data'] = vm_l1

    print(" # Computing similarity cost without offset")
    nominal_cost = compare_on_window(df_openpose, df_mocap, time_window=2, time_step=5)
    print(" similarity cost: {:.5f} \n".format(nominal_cost))

    # offset list to test
    offsets = [-1, -2, -3, 1, 2, 3, 4]
    print(" # Testing these offsets : {} \n".format(offsets))

    costs = []
    min_cost = nominal_cost
    best_offset = 0
    for offset in offsets:
        df_mocap['Time'] = (t_vm1 + offset)
        print(" Computing similarity cost for offset: {}".format(offset))
        c = compare_on_window(df_openpose, df_mocap, time_window=2, time_step=5)
        costs += [c]

        print(" Offset: {} --> similarity cost: {:.5f} \n".format(offset, c))

        if c < min_cost:
            min_cost = c
            best_offset = offset

    print(" # Best cost ({:.5f}) for offset: {}".format(min_cost, best_offset))

    fig, ax = plt.subplots(nrows=2, ncols=1)

    ax[0].plot(t_v, v_l1, 'g-', label='LH velocity 1')
    ax[0].plot(t_vm1, vm_l1, 'r-', label='LH velocity 1 - MOCAP')
    ax[0].legend(loc='upper left')

    ax[1].plot(t_v, v_l1, 'g-', label='LH velocity 1')
    ax[1].plot(t_vm1 + best_offset, vm_l1, 'r-', label='LH velocity 1 - MOCAP - shifted ({})'.format(best_offset))
    ax[1].legend(loc='upper left')

    # To maximize the plots
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()
