from __future__ import print_function

import os
import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from torchvision import transforms, datasets

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from shared_files.util import AverageMeter
from shared_files.util import adjust_learning_rate, warmup_learning_rate, accuracy
from shared_files.util import set_optimizer, save_model


from shared_files.PickleDataset import make_dataset
from models.model import WiFiAE, WifiEncoder
from torch.utils.data import ConcatDataset
import yaml
import pickle

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

# This collate_fn_padd will include the path of the data so we can generate a new dataset
def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''

    batch_data = {'modality': batch[0]['modality'],
                  'data_path': [sample['data_path'] for sample in batch],
                  'scene': [sample['scene'] for sample in batch],
                  'subject': [sample['subject'] for sample in batch],
                  'action': [sample['action'] for sample in batch],
                  'idx': [sample['idx'] for sample in batch] if 'idx' in batch[0] else None
                  }
    _output = [np.array(sample['output']) for sample in batch]
    _output = torch.FloatTensor(np.array(_output))
    batch_data['output'] = _output

    for mod in batch_data['modality']:
        if mod in ['mmwave', 'lidar']:
            _input = [torch.Tensor(sample['input_' + mod]) for sample in batch]
            _input = torch.nn.utils.rnn.pad_sequence(_input)
            _input = _input.permute(1, 2, 0, 3)
            batch_data['input_' + mod] = _input
        else:
            _input = [np.array(sample['input_' + mod]) for sample in batch]
            _input = torch.FloatTensor(np.array(_input))
            batch_data['input_' + mod] = _input

    return batch_data

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
    parser.add_argument('--learning_rate', type=float, default=1e-3,
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
    parser.add_argument('--data_folder', type=str, default='../../UTD-split-0507/UTD-split-0507-222-1/', help='data_folder')
    parser.add_argument('--reference_modality', type=str, default='depth',
                        choices=['depth', 'mmwave'], help='modality')
    parser.add_argument('--pair_metric', type=str, default='model_pretrain_AE', help='pair_metric')
    parser.add_argument('--num_class', type=int, default=27,
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
    parser.add_argument('--ckpt', type=str, default='./save_mmbind/save_train_AB_skeleton_AE/models/single_train_AB_lr_0.0001_decay_0.0001_bsz_64/',
                        help='path to pre-trained model')
    parser.add_argument('--seed', type=int, default=100)
    opt = parser.parse_args()
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    return opt

def set_dataset(opt):

    # Return two dataloaders, one for dataset A (depth) and dataset B (mmWave)
    print("train labeled data:")
    with open('../Configs/config_train_A.yaml', 'r') as handle:
            config_train = yaml.load(handle, Loader=yaml.FullLoader)
            train_dataset_A, _ = make_dataset('/mnt/ssd_8t/jason/MMFI_Dataset/', config_train)
    with open('../Configs/config_train_B.yaml', 'r') as handle:
            config_train = yaml.load(handle, Loader=yaml.FullLoader)
            train_dataset_B, _ = make_dataset('/mnt/ssd_8t/jason/MMFI_Dataset/', config_train)

   

    return train_dataset_A, train_dataset_B



def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_dataset_A, train_dataset_B = set_dataset(opt)

    # build model and criterion
    #model = set_model(opt)

    # Perform the validation for both datasets, paths_A, B used for creating new dataset
    # features_A, label_A, paths_A = validate(train_loader_A, model, opt)
    # features_B, label_B, paths_B = validate(train_loader_B, model, opt)

    if opt.reference_modality == "depth":
        reference_set = train_dataset_A
        search_set = train_dataset_B
    else:
        reference_set = train_dataset_B
        search_set = train_dataset_A

    save_paired_path = './save_mmbind/train_' + str(opt.reference_modality) + '_dataset/'
    correct_map = 0
    for sample_index in range(len(reference_set)):

        selected_index = np.random.randint(0, len(search_set))
        ref_sample = reference_set[sample_index]
        selected_sample = search_set[selected_index]
        if ref_sample['action'] == selected_sample['action']:
            correct_map += 1
            
        print(ref_sample['data_path'] + '\t' + ref_sample['action'] + '\t' + selected_sample['data_path'] + '\t' + 
              selected_sample['action'] + '\t' + str(ref_sample['action'] == selected_sample['action']))

        relative_path = save_paired_path + '/'.join(ref_sample['data_path'].split('/')[-3:])
        if not os.path.isdir(relative_path):
            os.makedirs(relative_path)
        with open(ref_sample['data_path'] + '/data.pickle', 'rb') as handle:
            ref_sample = pickle.load(handle)
        with open(selected_sample['data_path'] + '/data.pickle', 'rb') as handle:
            selected_sample = pickle.load(handle)    
        if opt.reference_modality == "depth":
            # Add in the mmWave data
            ref_sample['input_mmwave'] = selected_sample['input_mmwave']
        else:

            ref_sample['input_depth'] = selected_sample['input_depth']
  
        with open(relative_path + '/data.pickle', 'wb') as handle:
            pickle.dump(ref_sample, handle)
        
    print(correct_map, correct_map/len(reference_set), len(reference_set))



if __name__ == '__main__':
    main()