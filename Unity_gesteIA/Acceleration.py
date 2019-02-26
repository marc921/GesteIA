import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np

from utils import norm, get_velocity_norm, get_acc, get_acc_2, convolve_xy, dt, get_gaussian_kernel, get_velocity

# Parameters
width = 480
height = 270
openpose_files_path = os.path.join('.', 'OpenposeFiles')
scale_factor = 10000
left_hand_idx = 7
right_hand_idx = 4


def process_raw_list(l):
    l_rescale = [l[i] / scale_factor for i in range(len(l))]
    x = []
    y = []
    confidence = []
    if len(l) % 3 != 0:
        print("ERROR")
    else:
        for i in range(25):
            x += [l_rescale[i*3]]
            y += [l_rescale[i*3 + 1]]
            confidence += [l[i*3 + 2]]
    return x, y, confidence


def get_hands(l):
    x, y, _ = process_raw_list(l)
    return {"x": x[left_hand_idx], "y": y[left_hand_idx]}, {"x": x[right_hand_idx], "y": y[right_hand_idx]}


def process_folder(folder_path):
    json_files = [p for p in os.listdir(folder_path) if ".json" in p]
    person_1 = {
        "left_hand": {
            "x": [],
            "y": []
        },
        "right_hand": {
            "x": [],
            "y": []
        }
    }
    person_2 = {
        "left_hand": {
            "x": [],
            "y": []
        },
        "right_hand": {
            "x": [],
            "y": []
        }
    }

    for file in json_files:
        js = file = json.loads(open(os.path.join(folder_path, file), 'r').read())
        people_list = file['people']

        # Person 1
        l_hand, r_hand = get_hands(people_list[0]['pose_keypoints_2d'])
        person_1["left_hand"]["x"] += [l_hand["x"]]
        person_1["left_hand"]["y"] += [l_hand["y"]]
        person_1["right_hand"]["x"] += [r_hand["x"]]
        person_1["right_hand"]["y"] += [r_hand["y"]]

        # Person 2
        l_hand, r_hand = get_hands(people_list[1]['pose_keypoints_2d'])
        person_2["left_hand"]["x"] += [l_hand["x"]]
        person_2["left_hand"]["y"] += [l_hand["y"]]
        person_2["right_hand"]["x"] += [r_hand["x"]]
        person_2["right_hand"]["y"] += [r_hand["y"]]

    return person_1, person_2


# Process Openpose output files
person_1, person_2 = process_folder(openpose_files_path)

# Position plotting
t = [dt*i for i in range(len(person_1["left_hand"]["y"]))]

gaussian_kernel = get_gaussian_kernel(7)
x_rh1, y_rh1 = convolve_xy(person_1["right_hand"]["x"], person_1["right_hand"]["y"], gaussian_kernel)
x_lh1, y_lh1 = convolve_xy(person_1["left_hand"]["x"], person_1["left_hand"]["y"], gaussian_kernel)
x_rh2, y_rh2 = convolve_xy(person_2["right_hand"]["x"], person_2["right_hand"]["y"], gaussian_kernel)
x_lh2, y_lh2 = convolve_xy(person_2["left_hand"]["x"], person_2["left_hand"]["y"], gaussian_kernel)

plt.plot(t, y_rh1, 'r-', label='Right hand 1')
plt.plot(t, y_lh1, 'b-', label='Left hand 1')
plt.legend(loc='upper left')
plt.show()

# Velocity computing and plotting
v_r1 = get_velocity_norm(x_rh1, y_rh1)
v_l1 = get_velocity_norm(x_lh1, y_lh1)

v_r1_x, v_r1_y = get_velocity(x_rh1, y_rh1)
v_l1_x, v_l1_y = get_velocity(x_lh1, y_lh1)

t_v = [dt*i for i in range(len(v_r1))]

plt.plot(t_v, v_r1, 'r-', label='Right hand velocity 1')
plt.plot(t_v, v_l1, 'g-', label='Left hand velocity 1')
plt.legend(loc='upper left')
plt.show()

plt.plot(t_v, v_r1_y, 'r-', label='Right hand velocity 1')
plt.plot(t_v, v_l1_y, 'g-', label='Left hand velocity 1')
plt.legend(loc='upper left')
plt.show()

# Acceleration computing and plotting
a_r1 = get_acc(x_rh1, y_rh1)
a_l1 = get_acc(x_lh1, y_lh1)
a_r2 = get_acc(x_rh2, y_rh2)
a_l2 = get_acc(x_lh2, y_lh2)

t_a = [dt*i for i in range(len(a_r1))]

plt.plot(t_a, a_l1, 'r-', label='Left hand acceleration 1')
# plt.plot(t_a, a_l2, 'b-', label='Left hand acceleration 2')
plt.plot(t_a, a_r1, 'g-', label='Right hand acceleration 1')
# plt.plot(t_a, a_r2, 'p-', label='Right hand acceleration 2')
plt.legend(loc='upper left')
plt.show()

# Trying something
a_r1_x, a_r1_y = get_velocity(v_r1_x, v_r1_y)
a_l1_x, a_l1_y = get_velocity(v_l1_y, v_l1_y)

plt.plot(t_a, a_r1_y, 'r-', label='Right hand acceleration 1')
plt.plot(t_a, a_l1_y, 'g-', label='Left hand acceleration 1')
plt.legend(loc='upper left')
plt.show()
