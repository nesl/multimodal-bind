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
from models.model import SkeletonAE, SkeletonEncoder
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

def set_loader(opt):

    # Return two dataloaders, one for dataset A (depth) and dataset B (mmWave)
    print("train labeled data:")
    with open('../Configs/config_train_A.yaml', 'r') as handle:
            config_train = yaml.load(handle, Loader=yaml.FullLoader)
            train_dataset_A, _ = make_dataset('../../MMFI_Dataset/', config_train)
    with open('../Configs/config_train_B.yaml', 'r') as handle:
            config_train = yaml.load(handle, Loader=yaml.FullLoader)
            train_dataset_B, _ = make_dataset('../../MMFI_Dataset/', config_train)

    train_loader_A = torch.utils.data.DataLoader(
        train_dataset_A, batch_size=opt.batch_size, collate_fn=collate_fn_padd,
        num_workers=opt.num_workers, pin_memory=True, shuffle=False)

    train_loader_B = torch.utils.data.DataLoader(
        train_dataset_B, batch_size=opt.batch_size, collate_fn=collate_fn_padd,
        num_workers=opt.num_workers, pin_memory=True, shuffle=False)

    return train_loader_A, train_loader_B

def set_model(opt):

    model = SkeletonEncoder()
    # Use the skeletonAE() as a template to load weights in and then copy the weights
    model_template = SkeletonAE()
    model_template.load_state_dict(torch.load('./save_mmbind/save_train_AB_skeleton_AE/models/single_train_AB_lr_0.001_decay_0.0001_bsz_64/last.pth')['model'])
    model = model_template.skeleton_encoder
    model = model.cuda()
    return model



def validate(val_loader, model, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()

    confusion = np.zeros((opt.num_class, opt.num_class))
    features_list = []
    label_list = []
    idx_list = []

    with torch.no_grad():
        end = time.time()
        for idx, batched_data in enumerate(val_loader):
            # Labels in this case will be strings such as 'A01'
            labels = batched_data['action']


            bsz = len(labels)
            # batched_data contains the paths as well
            data_path = batched_data['data_path']
            # Get skeleton encoder features
            features = model(batched_data)

            # calculate and store confusion matrix
            features_list.extend(features.cpu().numpy())
            idx_list.extend(data_path)
            label_list.extend(labels)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                       idx, len(val_loader), batch_time=batch_time))

    features_list = np.array(features_list)
    idx_list = np.array(idx_list) # Holds all the data paths
    label_list = np.array(label_list)

    return features_list, label_list, idx_list


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader_A, train_loader_B = set_loader(opt)

    # build model and criterion
    model = set_model(opt)

    # Perform the validation for both datasets, paths_A, B used for creating new dataset
    features_A, label_A, paths_A = validate(train_loader_A, model, opt)
    features_B, label_B, paths_B = validate(train_loader_B, model, opt)

    if opt.reference_modality == "depth":
        ## dataset A as reference
        reference_feature = features_A
        reference_label = label_A # List of strings, start at 1
        search_feature = features_B
        search_label = label_B # List of strings, start at 1
    else:
        ## dataset B as reference
        reference_feature = features_B
        reference_label = label_B
        search_feature = features_A
        search_label = label_A

    save_paired_path = "./save_mmbind/train_{}_paired_AB/".format(opt.reference_modality)
    

    if not os.path.isdir(save_paired_path):
        os.makedirs(save_paired_path)

    similarity_matrix = cosine_similarity(reference_feature, search_feature)
    print(similarity_matrix)

    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Cosine Similarity Matrix - {}'.format(opt.pair_metric))
    if opt.reference_modality == "depth":
        plt.ylabel('Skeleton feature from Dataset A')
        plt.xlabel('Skeleton feature from Dataset B')
    else:
        plt.ylabel('Skeleton feature from Dataset B')
        plt.xlabel('Skeleton feature from Dataset A')
    # plt.show()
    #plt.savefig(save_paired_path + "similarity_matrix_{}_pair".format(opt.reference_modality))


    paired_data_length = len(reference_label)
    search_data_length = search_feature.shape[0]
    select_label = np.zeros(paired_data_length)
    correct_map = 0
    
    similarity_record = np.zeros(paired_data_length)


    for sample_index in range(paired_data_length):

        temp_similarity_vector = similarity_matrix[sample_index]

        select_feature_index = np.argmax(temp_similarity_vector)
        # Convert to a label of 1 to 26
        select_label[sample_index] = int(search_label[select_feature_index][1:])

        similarity_record[sample_index] = np.max(temp_similarity_vector)

        if reference_label[sample_index] == search_label[select_feature_index]:
            correct_map += 1
            temp_correct = 1
        else:
            temp_correct = 0


        print(reference_label[sample_index], search_label[select_feature_index], select_feature_index, np.max(temp_similarity_vector), temp_correct, paths_A[sample_index], paths_B[select_feature_index])

        # Get the reference data dictionary and the select data dictionary. 
        # Goal is to create a new field ['input_X'] containing data from select
        if opt.reference_modality == "depth":

            with open(paths_A[sample_index] + '/data.pickle', 'rb') as handle:
                reference_data = pickle.load(handle)
            with open(paths_B[select_feature_index] + '/data.pickle', 'rb') as handle:
                select_data = pickle.load(handle)

            # Save with the same environment/Subject/action folder structure, A and B have diff subjects
            relative_path_A = save_paired_path + '/'.join(paths_A[sample_index].split('/')[-3:])
            if not os.path.isdir(relative_path_A):
                os.makedirs(relative_path_A)
            # Add in the mmWave data
            reference_data['input_mmwave'] = select_data['input_mmwave']
            reference_data['similarity'] = similarity_record[sample_index]
            with open(relative_path_A + '/data.pickle', 'wb') as handle:
                pickle.dump(reference_data, handle)
        else:
            with open(paths_B[sample_index] + '/data.pickle', 'rb') as handle:
                reference_data = pickle.load(handle)
            with open(paths_A[select_feature_index] + '/data.pickle', 'rb') as handle:
                select_data = pickle.load(handle)

            relative_path_B = save_paired_path + '/'.join(paths_B[sample_index].split('/')[-3:])
            if not os.path.isdir(relative_path_B):
                os.makedirs(relative_path_B)
            reference_data['input_depth'] = select_data['input_depth']
            reference_data['similarity'] = similarity_record[sample_index]
            with open(relative_path_B + '/data.pickle', 'wb') as handle:
                pickle.dump(reference_data, handle)

    # np.save(save_paired_path + 'similarity.npy', similarity_record)
    # np.save(save_paired_path + 'label.npy', reference_label)

    print(similarity_record)
    print(correct_map, correct_map/paired_data_length)
    reference_label = [int(label[1:]) for label in reference_label]
    disp = ConfusionMatrixDisplay.from_predictions(reference_label, select_label)
    disp.plot() 
    if opt.reference_modality == "depth":
        disp.ax_.set_title("Pair Dataset A and B using Skeleton features")
        disp.ax_.set_ylabel('Label for Dataset A')
        disp.ax_.set_xlabel('Selected Label from Dataset B')
    else:
        disp.ax_.set_title("Pair Dataset B and A using Skeleton features")
        disp.ax_.set_ylabel('Label for Dataset B')
        disp.ax_.set_xlabel('Selected Label from Dataset A')
    #plt.savefig(save_paired_path + "results_{}_pair".format(opt.reference_modality))





if __name__ == '__main__':
    main()
