import os
import json
import matplotlib.pyplot as plt
import skimage.io

# Parameters
width = 480
height = 270


def process_raw_list(l):
    x = []
    y = []
    confidence = []
    if len(l) % 3 != 0:
        print("ERROR")
    else:
        for i in range(25):
            x += [l[i*3]]
            y += [l[i*3 + 1]]
            confidence += [l[i*3 + 2]]
    return x, y, confidence


def keep_good_points(x, y, confidence, thresh=0.8):
    x_good = []
    y_good = []

    for i in range(len(x)):
        if confidence[i] > thresh:
            x_good += [x[i]]
            y_good += [y[i]]

    return x_good, y_good


openpose_files_path = os.path.join('.', 'OpenposeFiles')
test_file = os.path.join(openpose_files_path, 'VR-FRFR13-14_compressed_down_4_000000000000_keypoints.json')
img_path = os.path.join(openpose_files_path, 'vlcsnap-00001.jpg')

file = json.loads(open(test_file, 'r').read())
people_list = file['people']
for people in people_list:
     raw_list = people['pose_keypoints_2d']
     x, y, confidence = process_raw_list(raw_list)
     x_good, y_good = keep_good_points(x, y, confidence, thresh=0.75)
     #plt.scatter(x_good, y_good)
     x_test = [x[0], x[4], x[7]]
     y_test = [y[0], y[4], y[7]]
     plt.scatter(x_test, y_test)

     plt.imshow(skimage.io.imread(img_path))
     plt.xlim(0, width)
     plt.ylim(height, 0)
     plt.show()
