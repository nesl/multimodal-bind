from __future__ import print_function

import os
import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
# from torchvision import transforms, datasets


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from shared_files.util import AverageMeter
from shared_files.util import adjust_learning_rate, warmup_learning_rate, accuracy
from shared_files.util import set_optimizer, save_model
from shared_files import data_pre as data

from models.acc_to_skeleton_gyro import MyUTDmodel_acc_to_skeleton, MyUTDmodel_acc_to_gyro

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
    parser.add_argument('--learning_rate', type=float, default=1e-4,
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
    parser.add_argument('--dataset', type=str, default='train_A',
                        choices=['train_A', 'train_B'], help='dataset')
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

    opt = parser.parse_args()

    # set the path according to the environment
    opt.save_path = './save_baseline4/save_{}_autoencoder/'.format(opt.dataset)
    opt.model_path = opt.save_path + 'models'
    opt.tb_path = opt.save_path + 'tensorboard'
    opt.result_path = opt.save_path + 'results/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'lr_{}_decay_{}_bsz_{}'.\
        format(opt.learning_rate, opt.weight_decay, opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)

    return opt


def set_loader(opt):

    #load labeled train and test data
    print("train data:")
    x_train_1, x_train_2, y_train = data.load_original_data(opt.dataset)# [Acc, Skeleton] or [Skeleton, Gyro]

    train_dataset = data.Multimodal_dataset(x_train_1, x_train_2, y_train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)

    return train_loader


def set_model(opt):

    if opt.dataset == "train_A":
        model = MyUTDmodel_acc_to_gyro(input_size=1)## generate gyro for dataset A
        opt.ckpt = "./save_baseline4/save_train_B_autoencoder/models/lr_0.0001_decay_0.0001_bsz_64/"
    else:
        model = MyUTDmodel_acc_to_skeleton(input_size=1)## generate skeleton for dataset A
        opt.ckpt = "./save_baseline4/save_train_A_autoencoder/models/lr_0.0001_decay_0.0001_bsz_64/"

    ckpt_path = opt.ckpt + 'last.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']


    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model = model.cuda()
        cudnn.benchmark = True

    model.load_state_dict(state_dict)

    return model




def validate(val_loader, model, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()

    features_list = []
    label_list = []
    original_data_list = []

    with torch.no_grad():
        end = time.time()
        for idx, (input_data1, input_data2, labels) in enumerate(val_loader):

            if torch.cuda.is_available():
                input_data1 = input_data1.cuda()
                input_data2 = input_data2.cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            features = model(input_data1).squeeze(1)
            print("generated data:", features.shape)


            # calculate and store confusion matrix
            features_list.extend(features.cpu().numpy())
            original_data_list.extend(input_data2.squeeze(1).cpu().numpy())
            label_list.extend(labels.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                       idx, len(val_loader), batch_time=batch_time))

    features_list = np.array(features_list)
    original_data_list = np.array(original_data_list)
    label_list = np.array(label_list)

    return features_list, original_data_list, label_list



def main():

    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model = set_model(opt)

    generate_data_list, original_data_list, label_list = validate(train_loader, model, opt)

    save_paired_path = "./save_baseline4/{}_generated_AB/".format(opt.dataset)
    if not os.path.isdir(save_paired_path):
        os.makedirs(save_paired_path + "skeleton/")
        os.makedirs(save_paired_path + "gyro/")

    for sample_index in range(len(label_list)):

        if opt.dataset == "train_A":
            np.save(save_paired_path + 'skeleton/{}.npy'.format(sample_index), original_data_list[sample_index])
            np.save(save_paired_path + 'gyro/{}.npy'.format(sample_index), generate_data_list[sample_index])
            np.save(save_paired_path + 'label.npy', label_list)
        else:
            np.save(save_paired_path + 'gyro/{}.npy'.format(sample_index), original_data_list[sample_index])
            np.save(save_paired_path + 'skeleton/{}.npy'.format(sample_index), generate_data_list[sample_index])
            np.save(save_paired_path + 'label.npy', label_list)
    

if __name__ == '__main__':
    main()
