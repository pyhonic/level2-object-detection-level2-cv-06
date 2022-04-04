import os
import cv2
import argparse

from mmdet.apis import init_detector, inference_detector

def main(args):
    data_dir = args.data_dir
    save_dir = args.save_dir
    image_dir = os.path.join(data_dir, 'train')
    image_list = sorted([
        os.path.join(image_dir, img) for img in os.listdir(image_dir)
    ])

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    config_file = args.config_file
    checkpoint_file = args.checkpoint
    device = 'cuda:0'

    model = init_detector(config_file, checkpoint_file, device=device)

    for image_path in image_list:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        result =  inference_detector(model, image_path)
        mask = result[1][0][0]
        for i in range(1, len(result[1][0])):
            mask = cv2.bitwise_or(mask, result[1][0][i])
        
        for i in range(3):
            img[:, :, i] = cv2.bitwise_or(mask, img[:, :, i], mask=mask)

        image_name = image_path.split('/')[-1]
        cv2.imwrite(
            os.path.join(save_dir, image_name, img)
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str)
    parser.add_argument('--save_dir', '-a', type=str)
    parser.add_argument('--config_file', '-c', type=str)
    parser.add_argument('--checkpoint', type=str)
    
    args = parser.parse_args()
    main(args)