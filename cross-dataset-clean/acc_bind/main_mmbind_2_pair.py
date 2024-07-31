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
# from torchvision import transforms, datasets

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
import data_pre as data

from single_modality import MyIMUmodel_acc_encoder


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

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

    # model dataset
    parser.add_argument('--model', type=str, default='MyUTDmodel')
    parser.add_argument('--dataset', type=str, default='Realworld', help='Realworld or Shoaib')
    parser.add_argument('--num_class', type=int, default=5,
                        help='num_class')
    parser.add_argument('--reference_modality', type=str, default='mag',
                        choices=['mag', 'gyro'], help='modality')
    parser.add_argument('--data_folder', type=str, default="../cross-split-data-0613/", help='data_folder')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--num_of_trial', type=int, default=1,
                        help='id for recording multiple runs')
    parser.add_argument('--ckpt', type=str, default='./save_mmbind/acc_autoencoder/models/lr_0.001_decay_0.0001_bsz_64/',#5e-4, bsz=64
                        help='path to pre-trained model')
    
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


def set_loader(opt):

    #load labeled train and test data
    x1_A, _, x3_A, y_A = data.load_data_IMU("Shoaib")#N=4500
    x1_B, x2_B, _, y_B = data.load_data_IMU("Motionsense")#N=12636

    train_dataset_A = data.Unimodal_dataset_idx(x1_A, y_A)
    train_dataset_B = data.Unimodal_dataset_idx(x1_B, y_B)

    train_loader_A = torch.utils.data.DataLoader(
        train_dataset_A, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    
    train_loader_B = torch.utils.data.DataLoader(
        train_dataset_B, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    
    return train_loader_A, train_loader_B


def set_model(opt):

    model = MyIMUmodel_acc_encoder(input_size=1)
    # criterion = torch.nn.CrossEntropyLoss()

    ckpt_path = opt.ckpt + 'last.pth'

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            if "acc_encoder" in k:
                new_state_dict[k] = v
        state_dict = new_state_dict
        model = model.cuda()
        cudnn.benchmark = True

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
        for idx, (data_idx, input_data, labels) in enumerate(val_loader):

            if torch.cuda.is_available():
                input_data = input_data.cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            # print("idx:", data_idx)
            features = model(input_data)
            # print("features:", features.shape)


            # calculate and store confusion matrix
            features_list.extend(features.cpu().numpy())
            idx_list.extend(data_idx)
            label_list.extend(labels.cpu().numpy())


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                       idx, len(val_loader), batch_time=batch_time))

    features_list = np.array(features_list)
    idx_list = np.array(idx_list)
    label_list = np.array(label_list)

    return features_list, label_list, idx_list


def main():

    opt = parse_option()

    train_loader_A, train_loader_B = set_loader(opt)

    # build model and criterion
    model = set_model(opt)
   
    features_A, label_A, idx_A = validate(train_loader_A, model, opt)#acc, mag
    features_B, label_B, idx_B = validate(train_loader_B, model, opt)#acc, gyro

    if opt.reference_modality == "mag":
        ## dataset A as reference
        reference_feature = features_A
        reference_label = label_A
        search_feature = features_B
        search_label = label_B
    else:
        ## dataset B as reference
        reference_feature = features_B
        reference_label = label_B
        search_feature = features_A
        search_label = label_A

    save_paired_path = "./save_mmbind/acc_paired_data_Motionsense_Shoaib/acc_remain_{}_pair/".format(opt.reference_modality)

    if not os.path.isdir(save_paired_path):
        os.makedirs(save_paired_path + "acc/")
        os.makedirs(save_paired_path + "gyro/")
        os.makedirs(save_paired_path + "mag/")

    similarity_matrix = cosine_similarity(reference_feature, search_feature)
    print(similarity_matrix)

    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Cosine Similarity Matrix')

    if opt.reference_modality == "mag":
        plt.ylabel('Acc feature from Dataset A')
        plt.xlabel('Acc feature from Dataset B')
    else:
        plt.ylabel('Acc feature from Dataset B')
        plt.xlabel('Acc feature from Dataset A')
    # plt.show()
    plt.savefig(save_paired_path + "similarity_matrix_{}_pair".format(opt.reference_modality))

    paired_data_length = reference_label.shape[0]
    search_data_length = search_feature.shape[0]
    select_label = np.zeros(paired_data_length)
    correct_map = 0
    
    similarity_record = np.zeros(paired_data_length)


    for sample_index in range(paired_data_length):

        temp_similarity_vector = similarity_matrix[sample_index]

        select_feature_index = np.argmax(temp_similarity_vector)
        select_label[sample_index] = search_label[select_feature_index]
        similarity_record[sample_index] = np.max(temp_similarity_vector)

        if reference_label[sample_index] == search_label[select_feature_index]:
            correct_map += 1
            temp_correct = 1
        else:
            temp_correct = 0

        print(reference_label[sample_index], search_label[select_feature_index], select_feature_index, np.max(temp_similarity_vector), temp_correct)

        if opt.reference_modality == "mag":
            ## dataset A as reference
            acc_data =  np.load(opt.data_folder + "Shoaib/acc/{}.npy".format(sample_index))
            mag_data = np.load(opt.data_folder + "Shoaib/mag/{}.npy".format(sample_index))
            gyro_data = np.load(opt.data_folder + "Motionsense/gyro/{}.npy".format(select_feature_index))

            acc_data = data.sensor_data_normalize_all('acc', acc_data, "Shoaib")
            mag_data = data.sensor_data_normalize_all('mag', mag_data, "Shoaib")
            gyro_data = data.sensor_data_normalize_all('gyro', gyro_data, "Motionsense")
            np.save(save_paired_path + 'acc/{}.npy'.format(sample_index), acc_data)
            np.save(save_paired_path + 'mag/{}.npy'.format(sample_index), mag_data)
            np.save(save_paired_path + 'gyro/{}.npy'.format(sample_index), gyro_data)
        else:
            ## dataset B as reference
            acc_data =  np.load(opt.data_folder + "Motionsense/acc/{}.npy".format(sample_index))
            gyro_data = np.load(opt.data_folder + "Motionsense/gyro/{}.npy".format(sample_index))
            mag_data = np.load(opt.data_folder + "Shoaib/mag/{}.npy".format(select_feature_index))

            acc_data = data.sensor_data_normalize_all('acc', acc_data, "Motionsense")
            gyro_data = data.sensor_data_normalize_all('gyro', gyro_data, "Motionsense")
            mag_data = data.sensor_data_normalize_all('mag', mag_data, "Shoaib")
            np.save(save_paired_path + 'acc/{}.npy'.format(sample_index), acc_data)
            np.save(save_paired_path + 'mag/{}.npy'.format(sample_index), mag_data)
            np.save(save_paired_path + 'gyro/{}.npy'.format(sample_index), gyro_data)

    np.save(save_paired_path + 'similarity.npy', similarity_record)
    np.save(save_paired_path + 'label.npy', reference_label)

    print(similarity_record)
    print(correct_map, correct_map/paired_data_length)

    disp = ConfusionMatrixDisplay.from_predictions(reference_label, select_label)
    disp.plot() 
    if opt.reference_modality == "acc":
        disp.ax_.set_title("Pair Dataset A and B using Acc features")
        disp.ax_.set_ylabel('Label for Dataset A')
        disp.ax_.set_xlabel('Selected Label from Dataset B')
    else:
        disp.ax_.set_title("Pair Dataset B and A using Acc features")
        disp.ax_.set_ylabel('Label for Dataset B')
        disp.ax_.set_xlabel('Selected Label from Dataset A')
    plt.savefig(save_paired_path + "results_{}_pair".format(opt.reference_modality))



if __name__ == '__main__':
    main()
