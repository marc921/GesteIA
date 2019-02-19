import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np

# Parameters
width = 480
height = 270
openpose_files_path = os.path.join('.', 'OpenposeFiles')
dt = 1/25


def process_raw_list(l):
    l_rescale = [l[i] / 10000 for i in range(len(l))]
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
    return {"x": x[7], "y": y[7]}, {"x": x[4], "y": y[4]}


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


person_1, person_2 = process_folder(openpose_files_path)

x = [dt*t for t in range(len(person_1["left_hand"]["y"]))]

# plt.plot(x, person_1["left_hand"]["y"])
# plt.show()


def norm(x, y):
    return math.sqrt(x**2 + y**2)


def get_velocity(x, y):
    v = []
    for i in range(len(x) - 1):
        vx = (x[i+1] - x[i]) / dt
        vy = (y[i+1] - y[i]) / dt
        v += [norm(vx, vy)]
    return v

person_1["right_hand"]["x"] = np.convolve(person_1["right_hand"]["x"], [0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006], mode='same')
person_1["right_hand"]["y"] = np.convolve(person_1["right_hand"]["y"], [0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006], mode='same')


v_left_1 = get_velocity(person_1["left_hand"]["x"], person_1["left_hand"]["y"])
v_right_1 = get_velocity(person_1["right_hand"]["x"], person_1["right_hand"]["y"])
v_left_2 = get_velocity(person_2["left_hand"]["x"], person_2["left_hand"]["y"])
v_right_2 = get_velocity(person_2["right_hand"]["x"], person_2["right_hand"]["y"])


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


acc_left_1 = get_acc(v_left_1)
acc_right_1 = get_acc(v_right_1)
acc_left_2 = get_acc(v_left_2)
acc_right_2 = get_acc(v_right_2)

x = [dt * i for i in range(len(acc_left_1))]

# plt.plot(x, acc_left_1)
# plt.plot(x, acc_right_1)
# plt.plot(x, acc_left_2)
# plt.plot(x, acc_right_2)

# plt.show()

acc_left_1 = get_acc_2(person_1["left_hand"]["x"], person_1["left_hand"]["y"])
acc_right_1 = get_acc_2(person_1["right_hand"]["x"], person_1["right_hand"]["y"])
acc_left_2 = get_acc_2(person_2["left_hand"]["x"], person_2["left_hand"]["y"])
acc_right_2 = get_acc_2(person_2["right_hand"]["x"], person_2["right_hand"]["y"])

plt.plot(x, acc_left_1)
plt.plot(x, acc_right_1)
# plt.plot(x, acc_left_2)
# plt.plot(x, acc_right_2)

#acc_diff = [acc_left_1[i] - acc_right_1[i] for i in range(len(acc_left_1))]
#plt.plot(x, acc_diff)

plt.show()
