# coding: utf-8

import glob
import os
from scipy.ndimage.filters import gaussian_filter
import cv2
import h5py
import numpy as np
import torch
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='CLTR')
parser.add_argument('--data_path', type=str, default='../../datasets/jhu_crowd_v2.0',
                    help='the data path of jhu')

args = parser.parse_args()
root = args.data_path

train = root + '/train/images/'
val = root + '/val/images/'
test = root + '/test/images/'

'''mkdir directories'''
if not os.path.exists(train.replace('images', 'images_2048')):
    os.makedirs(train.replace('images', 'images_2048'))
if not os.path.exists(train.replace('images', 'gt_detr_map_2048')):
    os.makedirs(train.replace('images', 'gt_detr_map_2048'))
if not os.path.exists(train.replace('images', 'gt_show')):
    os.makedirs(train.replace('images', 'gt_show'))

if not os.path.exists(val.replace('images', 'images_2048')):
    os.makedirs(val.replace('images', 'images_2048'))
if not os.path.exists(val.replace('images', 'gt_detr_map_2048')):
    os.makedirs(val.replace('images', 'gt_detr_map_2048'))
if not os.path.exists(val.replace('images', 'gt_show')):
    os.makedirs(val.replace('images', 'gt_show'))

if not os.path.exists(test.replace('images', 'images_2048')):
    os.makedirs(test.replace('images', 'images_2048'))
if not os.path.exists(test.replace('images', 'gt_detr_map_2048')):
    os.makedirs(test.replace('images', 'gt_detr_map_2048'))
if not os.path.exists(test.replace('images', 'gt_show')):
    os.makedirs(test.replace('images', 'gt_show'))

path_sets = [train, test, val]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()

for img_path in img_paths:

    img = cv2.imread(img_path)
    Img_data_pil = Image.open(img_path).convert('RGB')

    print(img_path)
    rate = 1
    rate1 = 1
    rate2 = 1
    if img.shape[1] >= img.shape[0] and img.shape[1] >= 2048:
        rate1 = 2048.0 / img.shape[1]
    elif img.shape[0] >= img.shape[1] and img.shape[0] >= 2048:
        rate1 = 2048.0 / img.shape[0]
    img = cv2.resize(img, (0, 0), fx=rate1, fy=rate1, interpolation=cv2.INTER_CUBIC)
    Img_data_pil = Img_data_pil.resize((img.shape[1], img.shape[0]) ,Image.ANTIALIAS)

    min_shape = 512.0
    if img.shape[1] <= img.shape[0] and img.shape[1] <= min_shape:
        rate2 = min_shape / img.shape[1]
    elif img.shape[0] <= img.shape[1] and img.shape[0] <= min_shape:
        rate2 = min_shape / img.shape[0]
    img = cv2.resize(img, (0, 0), fx=rate2, fy=rate2, interpolation=cv2.INTER_CUBIC)
    Img_data_pil = Img_data_pil.resize((img.shape[1], img.shape[0]), Image.ANTIALIAS)

    rate = rate1 * rate2

    k = np.zeros((img.shape[0], img.shape[1]))
    gt_file = np.loadtxt(img_path.replace('images', 'gt').replace('jpg', 'txt'))
    fname = img_path.split('/')[-1]

    try:
        y = gt_file[:, 0] * rate
        x = gt_file[:, 1] * rate
        for i in range(0, len(x)):
            if int(x[i]) < img.shape[0] and int(y[i]) < img.shape[1]:
                k[int(x[i]), int(y[i])] += 1
    except Exception:
        try:
            y = gt_file[0] * rate
            x = gt_file[1] * rate

            for i in range(0, 1):
                if int(x) < img.shape[0] and int(y) < img.shape[1]:
                    k[int(x), int(y)] += 1
        except Exception:
            ''' this image without person'''
            k = np.zeros((img.shape[0], img.shape[1]))

    kpoint = k.copy()
    kpoint = kpoint.astype(np.uint8)

    with h5py.File(img_path.replace('images', 'gt_detr_map_2048').replace('jpg', 'h5'), 'w') as hf:
        hf['kpoint'] = kpoint
        hf['image'] = Img_data_pil

    cv2.imwrite(img_path.replace('images', 'images_2048'), img)

print("end")
