import os
import json
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

# Settings
scale_factor = 1
left_hand_idx = 7
right_hand_idx = 4
height = 270
dt = 1/25

openpose_keypoints = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel"
]



def process_raw_list(l):
    l_rescale = [l[i] / scale_factor for i in range(len(l))]
    height_rescale = height / scale_factor
    x = []
    y = []
    confidence = []
    if len(l) % 3 != 0:
        print("ERROR")
    else:
        for i in range(25):
            x += [l_rescale[i*3]]
            if l_rescale[i*3 + 1] == 0:
                y += [0]
            else:
                y += [height_rescale - l_rescale[i*3 + 1]]
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


def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def export_to_csv(folder_path, write_to_disk=True, output_base_path='video_coordinates'):
    json_files = [p for p in os.listdir(folder_path) if ".json" in p]
    column_names = [name + "_x" for name in openpose_keypoints] + [name + "_y" for name in openpose_keypoints]
    data = np.zeros(shape=(len(json_files), len(column_names)))
    df1 = pd.DataFrame(data, columns=column_names, copy=True)
    df2 = pd.DataFrame(data, columns=column_names, copy=True)

    for cpt, file in tqdm(enumerate(json_files)):
        js = file = json.loads(open(os.path.join(folder_path, file), 'r').read())
        people_list = file['people']

        # Person 1
        x1, y1, confidence = process_raw_list(people_list[0]['pose_keypoints_2d'])

        # Person 2
        x2, y2, confidence = process_raw_list(people_list[1]['pose_keypoints_2d'])

        for i in range(len(openpose_keypoints)):
            d1 = dist(x1[i], y1[i], df1[openpose_keypoints[i] + "_x"].values[cpt-1], df1[openpose_keypoints[i] + "_y"].values[cpt-1])
            d2 = dist(x1[i], y1[i], df2[openpose_keypoints[i] + "_x"].values[cpt-1], df2[openpose_keypoints[i] + "_y"].values[cpt-1])

            if d1 < d2:
                df1[openpose_keypoints[i] + "_x"][cpt] = x1[i]
                df1[openpose_keypoints[i] + "_y"][cpt] = y1[i]

                df2[openpose_keypoints[i] + "_x"][cpt] = x2[i]
                df2[openpose_keypoints[i] + "_y"][cpt] = y2[i]
            else:
                df1[openpose_keypoints[i] + "_x"][cpt] = x2[i]
                df1[openpose_keypoints[i] + "_y"][cpt] = y2[i]

                df2[openpose_keypoints[i] + "_x"][cpt] = x1[i]
                df2[openpose_keypoints[i] + "_y"][cpt] = y1[i]

    t = [i*dt for i in range(len(json_files))]
    df1["time"] = t
    df1.to_csv(output_base_path + "_1.csv", index=False)

    df2["time"] = t
    df2.to_csv(output_base_path + "_2.csv", index=False)

    return df1, df2


def export_to_csv_tuples(folder_path, write_to_disk=True, output_base_path='video_coordinates_tuples'):
    json_files = [p for p in os.listdir(folder_path) if ".json" in p]
    data = np.zeros(shape=(len(json_files), len(openpose_keypoints)))
    dict1 = {}
    dict2 = {}

    for key in openpose_keypoints:
        dict1[key] = []
        dict2[key] = []

    for cpt, file in tqdm(enumerate(json_files)):
        js = file = json.loads(open(os.path.join(folder_path, file), 'r').read())
        people_list = file['people']

        # Person 1
        x1, y1, confidence = process_raw_list(people_list[0]['pose_keypoints_2d'])

        # Person 2
        x2, y2, confidence = process_raw_list(people_list[1]['pose_keypoints_2d'])
        for i in range(len(openpose_keypoints)):
            # if cpt > 0:
            #     if x1[i] == 0 and y1[i] == height:
            #         d1 = dist(x2[i], y2[i], dict2[openpose_keypoints[i]][-1][0], dict2[openpose_keypoints[i]][-1][1])
            #         d2 = dist(x2[i], y2[i], dict1[openpose_keypoints[i]][-1][0], dict1[openpose_keypoints[i]][-1][1])
            #     else:
            #         d1 = dist(x1[i], y1[i], dict1[openpose_keypoints[i]][-1][0], dict1[openpose_keypoints[i]][-1][1])
            #         d2 = dist(x1[i], y1[i], dict2[openpose_keypoints[i]][-1][0], dict2[openpose_keypoints[i]][-1][1])
            # else:
            #     d1 = 0
            #     d2 = 42

            if x1[i] < x2[i]:
                dict1[openpose_keypoints[i]] += [(x1[i], y1[i])]
                dict2[openpose_keypoints[i]] += [(x2[i], y2[i])]
            else:
                dict2[openpose_keypoints[i]] += [(x1[i], y1[i])]
                dict1[openpose_keypoints[i]] += [(x2[i], y2[i])]

    t = [i*dt for i in range(len(json_files))]
    df1 = pd.DataFrame(dict1)
    df1["time"] = t
    df1.to_csv(output_base_path + "_1.csv", index=False, sep=";")

    df2 = pd.DataFrame(dict2)
    df2["time"] = t
    df2.to_csv(output_base_path + "_2.csv", index=False, sep=";")

    return df1, df2


openpose_files_path = os.path.join('..', 'Unity_gesteIA', 'Assets', 'Data', 'OpenposeOutput', 'Down4')
output_path = os.path.join('..', 'Data', 'video_coordinates')
output_path_tuples = os.path.join('..', 'Data', 'video_coordinates_tuples')

# export_to_csv(openpose_files_path, output_base_path=output_path)
export_to_csv_tuples(openpose_files_path, output_base_path=output_path_tuples)
