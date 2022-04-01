import os
import json
import numpy as np
import argparse

def main(args):
    annotatio_path = os.path.join(args.data_dir, args.ann_file)
    save_path = os.path.join(args.data_dir, args.path)
    
    with open(annotatio_path) as f:
        json_file = json.loads(f.read())
        images = json_file['images']
        annotations = json_file['annotations']

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for x in ['small', 'medium', 'large']:
        if x == 'small':
            ann_data = [ann for ann in annotations if ann['area'] < 32**2]
        elif x == 'medium':
            ann_data = [ann for ann in annotations if 32**2 <= ann['area'] < 96**2]
        else:
            ann_data = [ann for ann in annotations if 96**2 <= ann['area']]

        image_id = np.unique([ann['image_id'] for ann in ann_data])
        img_data = [img for img in images if img['id'] in image_id]

        json_file['images'] = img_data
        json_file['annotations'] = ann_data

        with open(os.path.join(save_path, f'{args.ann_file.split(".")[0]}_{x}.json'), 'w') as f:
            json.dump(json_file, f)
    
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
        '--path', '-p', type=str, default='splitData'
    )
    args = parser.parse_args()
    main(args)