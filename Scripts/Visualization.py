import os
import json
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import time
import ast

# Parameters
width = 480
height = 270
openpose_label_x = 'LWrist_x'
openpose_label_y = 'LWrist_y'

# Process Openpose output files
data_path = os.path.join('..', 'Data')
person_1 = pd.read_csv(os.path.join(data_path, 'video_coordinates_tuples_1.csv'), sep=';')
person_2 = pd.read_csv(os.path.join(data_path, 'video_coordinates_tuples_2.csv'), sep=';')

# x_lh1 = person_1[openpose_label_x].values
# y_lh1 = person_1[openpose_label_y].values
# x_lh2 = person_2[openpose_label_x].values
# y_lh2 = person_2[openpose_label_y].values

person_1['LWrist'] = person_1['LWrist'].apply(ast.literal_eval)
T1 = person_1['LWrist'].values
x_lh1 = [e[0] for e in T1]
y_lh1 = [e[1] for e in T1]

person_2['LWrist'] = person_2['LWrist'].apply(ast.literal_eval)
T2 = person_2['LWrist'].values
x_lh2 = [e[0] for e in T2]
y_lh2 = [e[1] for e in T2]

cap = cv2.VideoCapture('VR-FRFR13-14_compressed_down_4.mp4')
cpt = 0

while(cap.isOpened()):
    ret, frame = cap.read()

    cv2.circle(frame, (int(x_lh1[cpt]), height - int(y_lh1[cpt])), 4, (255, 0, 0), -1)
    cv2.circle(frame, (int(x_lh2[cpt]), height - int(y_lh2[cpt])), 4, (0, 0, 255), -1)

    cpt += 1
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
