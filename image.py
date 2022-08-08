import cv2
import h5py
import numpy as np
from PIL import Image

def load_data(img_path, args, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_detr_map')

    while True:
        try:
            gt_file = h5py.File(gt_path)
            k = np.asarray(gt_file['kpoint'])
            img = np.asarray(gt_file['image'])
            img = Image.fromarray(img, mode='RGB')
            break
        except OSError:
            #print("path is wrong", gt_path)
            cv2.waitKey(1000)  # Wait a bit
    img = img.copy()
    k = k.copy()

    return img, k


def load_data_test(img_path, args, train=True):

    img = Image.open(img_path).convert('RGB')

    return img

