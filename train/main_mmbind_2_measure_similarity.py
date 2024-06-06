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
from shared_files import data_pre as data

from models.imu_models import SingleIMUEncoder, SingleIMUAutoencoder


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
    parser.add_argument('--reference_modality', type=str, default='setA',
                        choices=['gyro', 'mag', 'setA', 'setB'], help='modality')
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

    parser.add_argument('--common_modality', type=str, default="acc")

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    return opt


def set_loader(opt):

    mod = opt.common_modality
    mod_space = ['acc', 'gyro', 'mag']
    multi_mod_space = [[mod, m] for m in mod_space if m != mod]

    opt.valid_mod = multi_mod_space

    #load labeled train and test data
    print(f"=\tInitializing Dataset A for mod {multi_mod_space[0]}")
    train_datasetA = data.Multimodal_dataset([], multi_mod_space[0], root='../PAMAP_Dataset/trainA/')
    print(f"=\tInitializing Dataset B for mod {multi_mod_space[1]}")
    train_datasetB = data.Multimodal_dataset([], multi_mod_space[1], root='../PAMAP_Dataset/trainB/')

    train_loader_A = torch.utils.data.DataLoader(
        train_datasetA, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=False)

    train_loader_B = torch.utils.data.DataLoader(
        train_datasetB, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=False)

    return train_loader_A, train_loader_B



def set_model(opt):

    mod = opt.common_modality
    print(f"=\tLoading Autoencoder for modality {mod}")
    model = SingleIMUEncoder(mod)
    model_template= SingleIMUAutoencoder(mod)

    if mod == "acc":
        weight = "./save_mmbind/save_train_AB_acc_autoencoder_no_load/models/lr_0.005_decay_0.0001_bsz_64"
    elif mod == "gyro":
        weight = "./save_mmbind/save_train_AB_unimod_autoencoder_no_load_gyro/models/lr_0.0001_decay_0.0001_bsz_64"
    elif mod == "mag":
        weight = "./save_mmbind/save_train_AB_unimod_autoencoder_no_load_mag/models/lr_0.0001_decay_0.0001_bsz_64"
    else:
        raise Exception(f"Invalid modality {mod}")
    model_template.load_state_dict(torch.load(os.path.join(weight, "last.pth"))['model'])
    model = model_template.encoder
    model.cuda()
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

            
            labels = batched_data['action']
            data_paths = batched_data['data_path']

            bsz = labels.shape[0]

            # forward
            features = torch.reshape(model(batched_data), (bsz, -1))

            # calculate and store confusion matrix
            features_list.extend(features.cpu().numpy())
            idx_list.extend(data_paths)
            label_list.extend(labels.cpu().numpy())


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    features_list = np.array(features_list)
    idx_list = np.array(idx_list)
    label_list = np.array(label_list)

    return features_list, label_list, idx_list


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader_A, train_loader_B = set_loader(opt)

    # build model and criterion
    model = set_model(opt) # 

    features_A, label_A, paths_A = validate(train_loader_A, model, opt) # get features of X_c from D1
    features_B, label_B, paths_B = validate(train_loader_B, model, opt) # get features of X_c from D2

    mod_A = opt.valid_mod[0][1]
    mod_B = opt.valid_mod[1][1]

    print(f"=\tDataset A has mod: {mod_A}")
    print(f"=\tDataset B has mod: {mod_B}")

    print(f"=\tCommon Modality: {opt.common_modality}")
    print(f"=\tReference set: {opt.reference_modality}")
    
    if opt.reference_modality == "setA":
        ## dataset A as reference
        reference_feature = features_A
        reference_label = label_A
        reference_path = paths_A

        search_feature = features_B
        search_label = label_B
        search_path = paths_B
    else:
        ## dataset B as reference
        reference_feature = features_B
        reference_label = label_B
        reference_path = paths_B

        search_feature = features_A
        search_label = label_A
        search_path = paths_A

    save_paired_path = f"./save_mmbind/train_{opt.reference_modality}_paired_AB_{opt.common_modality}/"

    if not os.path.isdir(save_paired_path):
        os.makedirs(save_paired_path)

    similarity_matrix = cosine_similarity(reference_feature, search_feature)
    print(similarity_matrix)

    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Cosine Similarity Matrix - {}'.format(opt.pair_metric))

    if opt.reference_modality == "setA":
        plt.ylabel(f'{opt.common_modality} feature from Dataset A')
        plt.xlabel(f'{opt.common_modality} feature from Dataset B')
    else:
        plt.ylabel(f'{opt.common_modality} feature from Dataset B')
        plt.xlabel(f'{opt.common_modality} feature from Dataset A')
    # plt.show()
    plt.savefig(save_paired_path + f"similarity_matrix_{opt.reference_modality}_pair_{opt.common_modality}.png")

    paired_data_length = reference_label.shape[0]
    search_data_length = search_feature.shape[0]
    select_label = np.zeros(paired_data_length)
    correct_map = 0
    
    similarity_record = np.zeros(paired_data_length)

    file_counter = 0
    label_mapper = [1, 3, 4, 12, 13, 16, 17]
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

        index = {
            "acc": 7,
            "gyro": 10,
            "mag": 13,
        }
        reference_data = np.load(reference_path[sample_index])
        select_data = np.load(search_path[select_feature_index])
        

        # if reference is A, then select is B
        select_index_begin = index[mod_B] if opt.reference_modality == "setA" else index[mod_A]
        reference_data[:, select_index_begin:select_index_begin+3] = select_data[:, select_index_begin:select_index_begin+3] 
        np.save(save_paired_path + '/' +  str(label_mapper[label_A[sample_index]]) + f'_{opt.reference_modality}_' + str(file_counter) + '.npy', reference_data)

        file_counter += 1

    # np.save(save_paired_path + 'similarity.npy', similarity_record)
    # np.save(save_paired_path + 'label.npy', reference_label)

    print(similarity_record)
    print(correct_map, correct_map/paired_data_length)

    disp = ConfusionMatrixDisplay.from_predictions(reference_label, select_label)
    disp.plot() 
    if opt.reference_modality == "setA":
        disp.ax_.set_title(f"Pair Dataset A and B using {mod_A} features")
        disp.ax_.set_ylabel('Label for Dataset A')
        disp.ax_.set_xlabel('Selected Label from Dataset B')
    else:
        disp.ax_.set_title(f"Pair Dataset B and A using {mod_B} features")
        disp.ax_.set_ylabel('Label for Dataset B')
        disp.ax_.set_xlabel('Selected Label from Dataset A')
    plt.savefig(save_paired_path + f"results_{opt.reference_modality}_pair_{opt.common_modality}.png")





if __name__ == '__main__':
    main()
