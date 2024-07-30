from __future__ import print_function

import os
import sys
import argparse
import time
import math
import random

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from torchvision import transforms, datasets

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from shared_files.util import AverageMeter
from shared_files.util import adjust_learning_rate, warmup_learning_rate, accuracy
from shared_files.util import set_optimizer, save_model
from shared_files import data_pre as data
from models.fuse_2M_skeleton_gyro import MyUTDmodel_2M


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
    parser.add_argument('--epochs', type=int, default=200,
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
    parser.add_argument('--dataset', type=str, default='train_C/label_216/', help='dataset')
    parser.add_argument('--reference_modality', type=str, default='all', help='reference_modality')
    parser.add_argument('--contrastive', type=str, default='', help='_weighted or blank')
    parser.add_argument('--load_pretrain', type=str, default='load_pretrain', help='load_pretrain or no_pretrain')
    parser.add_argument('--num_class', type=int, default=27,
                        help='num_class')


    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--num_of_trial', type=int, default=5,
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    return opt


def set_seed(seed):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Set the seed for numpy
    np.random.seed(seed)
    
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # Ensure deterministic behavior in PyTorch, might affect the performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Set the seed for other potential sources of randomness
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def label_pair_data(opt):

    #load labeled train and test data
    print("train labeled data:")
    x1_A, y_A = data.load_single_dataset("train_A")
    x2_B, y_B = data.load_single_dataset("train_B")

    count_class = np.zeros((opt.num_class, 2))
    index_A = []
    index_B = []

    for class_id in range(opt.num_class):
        count_class[class_id, 0] = np.count_nonzero(y_A == class_id)
        count_class[class_id, 1] = np.count_nonzero(y_B == class_id)

        index_A.append(np.where(y_A == class_id)[0])
        index_B.append(np.where(y_B == class_id)[0])

    print(y_A.shape, y_B.shape, count_class)
    # print(index_A, index_B)

    for class_id in range(opt.num_class):
        print(index_A[class_id].shape)
        print(index_B[class_id].shape)

    x1 = []
    x2 = []
    y = []

    for class_id in range(opt.num_class):#opt.num_class

        for sample_id in range(index_A[class_id].shape[0]):

            sample_index = index_A[class_id][sample_id]
            # print(y_A[sample_index])

            for random_id in range(index_B[class_id].shape[0]):
                random_index = index_B[class_id][random_id]
                x1.append(x1_A[sample_index])## skeleton
                y.append(y_A[sample_index])
                x2.append(x2_B[random_index])#acc
            
        for sample_id in range(index_B[class_id].shape[0]):

            sample_index = index_B[class_id][sample_id]
            # print(y_B[sample_index])

            for random_id in range(index_A[class_id].shape[0]):
                random_index = index_A[class_id][random_id]
                x2.append(x2_B[sample_index])## gyro
                y.append(y_B[sample_index])
                x1.append(x1_A[random_index])#skeleton

    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    print(y, x1.shape, x2.shape, y.shape)

    folder_path = "./save_mmbind_more_label_paired_data/"#acc_remain
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    np.save(folder_path + 'skeleton.npy', x1)
    np.save(folder_path + 'gyro.npy', x2)
    np.save(folder_path + 'label.npy', y)


def main():

    opt = parse_option()

    label_pair_data(opt)


if __name__ == '__main__':
    main()
