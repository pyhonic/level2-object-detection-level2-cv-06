import pandas as pd
import numpy as np
import os

BASE = '/opt/ml/detection/baseline/yolo/yolov5/runs/detect/epoch100/labels'

def bbox(bbox_info):
    class_num = bbox_info[0]
    x_center = np.float(bbox_info[1])
    y_center = np.float(bbox_info[2])
    w = np.float(bbox_info[3])
    h = np.float(bbox_info[4])
    score = np.float(bbox_info[5])

    dw = w * 1024
    dh = h * 1024
    
    x = x_center * 1024
    y = y_center * 1024

    dx_min = x - dw / 2
    dy_min = y - dh / 2

    dx_max = dx_min + dw
    dy_max = dy_min + dh

    return [str(class_num), str(score), str(dx_min), str(dy_min), str(dx_max), str(dy_max)]


coor_list = os.listdir(BASE)

# print(len(coor_list))
box_info = []
voc_bbox = []

for i in coor_list:
    # print(i)
    temp = os.path.join(BASE, i)
    temp_box = []
    with open(temp) as file:
        lines = file.readlines()
        # print(lines)
        for line in lines:
            line = line.strip()
            # print(line)
            temp_box.append(line)
    
    box_info.append((temp_box, f'test/{i[:4]}.jpg'))
        
        # box_info.append([file.readlines(), f'test/{i[:4]}.jpg'])
box_info.sort(key=lambda x: x[1])

prediction_string = []
img_id = []

df = pd.DataFrame()
for info in box_info:
    # info[0]
    img_id.append(info[1])
    # print(info[0])
    pred_str = ''
    for bbox_info in info[0]:
        temp = bbox(bbox_info.split())
        for t in temp:
            pred_str += t + ' '
    prediction_string.append(pred_str)
    
df['PredictionString'] = prediction_string
df['image_id'] = img_id

df.to_csv('yolo_submission2.csv', index=False)
