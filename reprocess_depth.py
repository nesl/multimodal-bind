import os
import argparse

import yaml
import numpy as np
import torch
import pickle
import cv2


dataset_root = '/home/jason/Documents/MMBind_MMFI/MMFI_Dataset_backup/'
for root, dirs, files in os.walk(dataset_root):
    for file in files:
        if (file == 'data.pickle'):
            file_path = root + '/' + file
            with open(file_path, 'rb') as handle:
                data = pickle.load(handle)
            depth_data = []
            for depth_file in os.listdir(root + '/depth/'):
                depth_img = cv2.imread(root + '/depth/' + depth_file)
                depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
                depth_img = cv2.resize(depth_img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
                depth_data.append(depth_img)
            depth_data = np.stack(depth_data, axis=0)
            data['input_depth'] = depth_data
            with open(root + '/' + 'data.pickle', 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

