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
from shared_files.data_pre import TrainA_Lazy, TrainB_Lazy

from models.models import ImageAE
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


def set_loader(opt):

    #load labeled train and test data
    print("train labeled data:")
    train_dataset_A = TrainA_Lazy()
    train_dataset_B = TrainB_Lazy()

    train_loader_A = torch.utils.data.DataLoader(
        train_dataset_A, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=False)

    train_loader_B = torch.utils.data.DataLoader(
        train_dataset_B, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=False)

    return train_loader_A, train_loader_B, train_dataset_A, train_dataset_B



def set_model(opt):
    model = ImageAE()

    ckpt_path = opt.ckpt + 'ckpt_epoch_100.pth'

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        model = model.cuda()

    if opt.pair_metric == "model_pretrain_AE":
        model.load_state_dict(state_dict)
        
    model = model.enc # only use encoder

    return model




def validate(val_loader, model, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()

    features_list = []
    label_list = []

    with torch.no_grad():
        end = time.time()
        for idx, batched_data in enumerate(val_loader):

            if torch.cuda.is_available():
                for key in batched_data.keys():
                    batched_data[key] = batched_data[key].cuda()
            bsz = len(batched_data['label'])

            # forward
            print("idx:", idx)
            features = torch.reshape(model(batched_data), (bsz, -1)) # Pass image through encoder
            print("features:", features.shape)


            # calculate and store confusion matrix
            features_list.extend(features.cpu().numpy())
            label_list.extend(batched_data['label'].cpu().numpy())


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                       idx, len(val_loader), batch_time=batch_time))

    features_list = np.array(features_list)
    label_list = np.array(label_list)

    return features_list, label_list


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader_A, train_loader_B, train_dataset_A, train_dataset_B = set_loader(opt)

    # build model and criterion
    model = set_model(opt)

    features_A, label_A = validate(train_loader_A, model, opt)
    features_B, label_B = validate(train_loader_B, model, opt)

    if opt.reference_modality == "semseg":
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

    save_paired_path = "./save_mmbind/train_{}_paired_AB_test/".format(opt.reference_modality)

    if not os.path.isdir(save_paired_path):
        os.makedirs(save_paired_path + "semseg/")
        os.makedirs(save_paired_path + "depth/")
    similarity_matrix = cosine_similarity(reference_feature, search_feature)
    print(similarity_matrix)

    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Cosine Similarity Matrix - {}'.format(opt.pair_metric))
    if opt.reference_modality == "semseg":
        plt.ylabel('Img feature from Dataset A')
        plt.xlabel('Img feature from Dataset B')
    else:
        plt.ylabel('Img feature from Dataset B')
        plt.xlabel('Img feature from Dataset A')
    # plt.show()
    plt.savefig(save_paired_path + "similarity_matrix_{}_pair".format(opt.reference_modality))

    paired_data_length = reference_label.shape[0]
    select_label = np.zeros(paired_data_length)
    correct_map = 0
    
    similarity_record = np.zeros(paired_data_length)
    pairing_record = np.zeros(paired_data_length)


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

        pairing_record[sample_index] = temp_correct


        print(reference_label[sample_index], search_label[select_feature_index], select_feature_index, np.max(temp_similarity_vector), temp_correct)

        if opt.reference_modality == "semseg":
            ## dataset A as reference
            reference_data = train_dataset_A[sample_index]
            other_mod_data = train_dataset_B[select_feature_index]
            reference_data['depth'] = other_mod_data['depth']
            reference_data['similarity'] = similarity_record[sample_index]
            with open(save_paired_path + str(sample_index) + '.pickle', 'wb') as handle:
                pickle.dump(reference_data, handle)
        else:
            ## dataset B as reference
            reference_data = train_dataset_B[sample_index]
            other_mod_data = train_dataset_A[select_feature_index]
            reference_data['semseg'] = other_mod_data['semseg']
            reference_data['similarity'] = similarity_record[sample_index]
            with open(save_paired_path + str(sample_index) + '.pickle', 'wb') as handle:
                pickle.dump(reference_data, handle)

    np.save(save_paired_path + 'similarity.npy', similarity_record)
    np.save(save_paired_path + 'label.npy', reference_label)
    np.save(save_paired_path + 'pairing_record.npy', pairing_record)

    print(similarity_record)
    print(correct_map, correct_map/paired_data_length)

    disp = ConfusionMatrixDisplay.from_predictions(reference_label, select_label)
    disp.plot() 
    if opt.reference_modality == "semseg":
        disp.ax_.set_title("Pair Dataset A and B using Img features")
        disp.ax_.set_ylabel('Label for Dataset A')
        disp.ax_.set_xlabel('Selected Label from Dataset B')
    else:
        disp.ax_.set_title("Pair Dataset B and A using Img features")
        disp.ax_.set_ylabel('Label for Dataset B')
        disp.ax_.set_xlabel('Selected Label from Dataset A')
    plt.savefig(save_paired_path + "results_{}_pair".format(opt.reference_modality))





if __name__ == '__main__':
    main()
