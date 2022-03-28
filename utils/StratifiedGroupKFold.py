
import os
import json 
import argparse
import numpy as np 

from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold 


def main(args):
    annotation_path = os.path.join(args.data_dir, args.ann_file)
    save_path = os.path.join(args.data_dir, args.path)
    with open(annotation_path, 'r') as f:
        train_json = json.loads(f.read())
        images = train_json['images']
        annotations = train_json['annotations']

    var = [(ann['image_id'], ann['category_id']) for ann in annotations]
    X = np.ones((len(annotations),1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var]) 

    cv = StratifiedGroupKFold(n_splits=args.n_split, shuffle=True, random_state=args.seed) 

    if not os.path.exists(save_path):
        os.mkdir(save_path)


    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)): 
        train_dict = defaultdict(list)
        val_dict = defaultdict(list)
        for x in ['info', 'licenses', 'categories']:
            train_dict[x].extend(train_json[x])
            val_dict[x].extend(train_json[x])

        for image in images:
            image_id = image['id']
            if image_id in train_idx:
                train_dict['images'].append(image)
            else:
                val_dict['images'].append(image)
        
        for annotation in annotations:
            image_id = annotation['image_id']
            if image_id in train_idx:
                train_dict['annotations'].append(annotation)
            else:
                val_dict['annotations'].append(annotation)

        with open(os.path.join(save_path, f"train_{fold}.json"), 'w') as train_file:
            json.dump(train_dict, train_file)
        with open(os.path.join(save_path, f"val_{fold}.json"), 'w') as val_file:
            json.dump(val_dict, val_file)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', '-d', type=str, default='/opt/ml/detection/dataset',
        help='data directory'
    )
    parser.add_argument(
        '--ann_file', '-a', type=str, default='train.json',
        help='annotation file'
    )
    parser.add_argument(
        '--n_split', '-n', type=int, default=5,
    )
    parser.add_argument(
        '--path', '-p', type=str, default='stratified_kfold'
    )
    parser.add_argument(
        '--seed', '-s', type=int, default=42
    )
    args = parser.parse_args()
    main(args)