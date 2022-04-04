import os
import json
import copy

import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm

from ensemble_boxes import *
from pycocotools.coco import COCO


def main(args):
    anno_train = args.anno_train 
    anno_test = args.anno_test
    save_path = args.save_path
    csv_path = args.csv_file
    iou_thr = args.iou_threshold
    conf_thr = args.conf_threshold

    with open(anno_train, 'r') as train_json:
        train_data = json.loads(train_json.read())
        train_anno = train_data['annotations']
        train_img = train_data['images']

    with open(anno_test, 'r') as test_json:
        test_data = json.loads(test_json.read())
        test_img = test_data['images']

    pseudo_dict = {
        'info': train_data['info'].copy(),
        'licenses': train_data['licenses'].copy(),
        'images': train_img.copy(),
        'categories': train_data['categories'].copy(),
        'annotations': train_anno.copy()
    }

    anno_count = len(train_anno)
    image_count = len(train_img)

    df = pd.read_csv(csv_path)


    predictions = df['PredictionString'].tolist()
    image_ids = df['image_id'].tolist()    
    
    anno_list = []
    img_list = []

    for idx, image_id in tqdm(enumerate(image_ids)):
        boxes_list = []
        scores_list = []
        labels_list = []
        image_info = test_img[idx]
        
        prediction_list = str(predictions[idx]).split()

        if len(prediction_list) == 0 or len(prediction_list) == 1:
            continue 

        prediction_list = np.reshape(prediction_list, (-1, 6))
        box_list = []
        
        for box in prediction_list[:, 2:6].tolist():
            box[0] = float(box[0]) / image_info['width']
            box[1] = float(box[1]) / image_info['height']
            box[2] = float(box[2]) / image_info['width']
            box[3] = float(box[3]) / image_info['height']
            box_list.append(box)

        boxes_list.append(box_list)
        scores_list.append(list(map(float, prediction_list[:, 1].tolist())))
        labels_list.append(list(map(int, prediction_list[:, 0].tolist())))
        
        if len(boxes_list):
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=iou_thr, skip_box_thr=conf_thr)

            for box, score, label in zip(boxes, scores, labels):
                x_min, y_min = box[0] * image_info['height'], box[1] * image_info['width']
                width, height = (box[2] - box[0]) * image_info['height'], (box[3] - box[1]) * image_info['width']
                anno_list.append({
                    'image_id': image_count,
                    'category_id': int(label),
                    'area': width * height,
                    "bbox": [x_min, y_min, width, height],
                    'iscrowd': 0,
                    'id': anno_count
                })
                anno_count += 1

            image_dict = {
                key:value for key, value in image_info.items()
            }
            image_dict['id'] = image_count
            img_list.append(image_dict)
            image_count += 1

        pseudo_dict['images'].extend(img_list)
        pseudo_dict['annotations'].extend(anno_list)

    with open(os.path.join(save_path, 'temp.json'), 'w') as f:
        json.dump(pseudo_dict, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='/opt/ml/detection/EDA/output.csv',
                        help='csv file path')
    parser.add_argument('--save_path', type=str, default='/opt/ml/detection/dataset/')
    parser.add_argument('--anno_train', type=str, default='/opt/ml/detection/dataset/train.json')
    parser.add_argument('--anno_test', type=str, default='/opt/ml/detection/dataset/test.json')
    parser.add_argument('--conf_threshold', type=float, default=0.3)
    parser.add_argument('--iou_threshold', type=float, default=0.5)

    args = parser.parse_args()
    main(args)