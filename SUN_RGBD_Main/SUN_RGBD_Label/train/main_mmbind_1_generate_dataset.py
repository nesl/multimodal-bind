from __future__ import print_function

import os
import sys
import argparse
import time
import math
from pathlib import Path

# import tensorboard_logger as tb_logger
import random
# from torchvision import transforms, datasets

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

from shared_files.util import adjust_learning_rate, warmup_learning_rate, accuracy
from shared_files.util import set_optimizer, save_model
from shared_files.data_pre import TrainA_Lazy, TrainB_Lazy

from collections import defaultdict
import pickle


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--num_positive', type=int, default=2,
                        help='number of positive samples')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # model dataset
    parser.add_argument('--model', type=str, default='MyUTDmodel')
    parser.add_argument('--data_folder', type=str, default='../../UTD-split-0507/UTD-split-0507-222-4/', help='data_folder')
    parser.add_argument('--reference_modality', type=str, default='semseg',
                        choices=['semseg', 'depth'], help='modality')
    parser.add_argument('--pair_metric', type=str, default='model_pretrain_AE', help='pair_metric')
    parser.add_argument('--num_class', type=int, default=6,
                        help='num_class')


    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--ckpt', type=str, default='./save_mmbind/save_train_AB_acc_AE/models/single_train_AB_lr_0.0001_decay_0.0001_bsz_64/',
                        help='path to pre-trained model')

    opt = parser.parse_args()


    return opt




def main():
    train_A = TrainA_Lazy()
    train_B = TrainB_Lazy()

    A_dict = defaultdict(list)
    B_dict = defaultdict(list)

    save_a_path = './save_mmbind/save_train_A/'
    save_b_path = './save_mmbind/save_train_B/'

    Path(save_a_path).mkdir(parents=True, exist_ok=True)
    Path(save_b_path).mkdir(parents=True, exist_ok=True)

    for item in train_A:
        A_dict[item['label']].append(item)
    for item in train_B:
        B_dict[item['label']].append(item)
    
    idx = 0

    for key in A_dict.keys():
        print("A_dict", key)
        B_arr = B_dict[key]
        for item in A_dict[key]:
            rand_index = int(random.random() * len(B_arr))
            item['depth'] = B_arr[rand_index]['depth']
            with open(save_a_path + str(idx) + '.pickle', 'wb') as handle:
                pickle.dump(item, handle)
                idx += 1
    idx = 0
    for key in B_dict.keys():
        print("B_dict", key)
        A_arr = A_dict[key]
        for item in B_dict[key]:
            rand_index = int(random.random() * len(A_arr))
            item['semseg'] = A_arr[rand_index]['semseg']
            with open(save_b_path + str(idx) + '.pickle', 'wb') as handle:
                pickle.dump(item, handle)
                idx += 1




if __name__ == '__main__':
    main()
