import os
import numpy as np
import argparse

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')

'''please set your dataset path'''

parser = argparse.ArgumentParser(description='CLTR')
parser.add_argument('--jhu_path', type=str, default='../datasets/jhu_crowd_v2.0',
                    help='the data path of jhu')
parser.add_argument('--nwpu_path', type=str, default='../datasets/NWPU_CLTR',
                    help='the data path of jhu')

args = parser.parse_args()
jhu_root = args.jhu_path
nwpu_root = args.nwpu_path

try:

    Jhu_train_path = jhu_root + '/train/images_2048/'
    Jhu_val_path = jhu_root + '/val/images_2048/'
    jhu_test_path = jhu_root + '/test/images_2048/'

    train_list = []
    for filename in os.listdir(Jhu_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(Jhu_train_path + filename)
    train_list.sort()
    np.save('./npydata/jhu_train.npy', train_list)

    val_list = []
    for filename in os.listdir(Jhu_val_path):
        if filename.split('.')[1] == 'jpg':
            val_list.append(Jhu_val_path + filename)
    val_list.sort()
    np.save('./npydata/jhu_val.npy', val_list)

    test_list = []
    for filename in os.listdir(jhu_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(jhu_test_path + filename)
    test_list.sort()
    np.save('./npydata/jhu_test.npy', test_list)

    print("Generate JHU image list successfully", len(train_list), len(val_list), len(test_list))
except:
    print("The JHU dataset path is wrong. Please check your path.")


try:
    f = open("./data/NWPU_list/train.txt", "r")
    train_list = f.readlines()

    f = open("./data/NWPU_list/val.txt", "r")
    val_list = f.readlines()

    '''nwpu dataset path'''
    root = nwpu_root + '/gt_detr_map/'


    if not os.path.exists(root):
        print("The NWPU dataset path is wrong. Please check your path.")

    else:
        train_img_list = []
        for i in range(len(train_list)):
            fname = train_list[i].split(' ')[0] + '.jpg'
            train_img_list.append(root + fname)


        val_img_list = []
        for i in range(len(val_list)):
            fname = val_list[i].split(' ')[0] + '.jpg'
            val_img_list.append(root + fname)

        np.save('./npydata/nwpu_train.npy', train_img_list)
        np.save('./npydata/nwpu_val.npy', val_img_list)


        print("Generate NWPU image list successfully", len(train_img_list), len(val_img_list))
except:
    print("The NWPU dataset path is wrong. Please check your path.")

