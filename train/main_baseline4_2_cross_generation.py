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

from models.imu_models import SingleIMUAutoencoder
from modules.option_utils import parse_option
from modules.print_utils import pprint

from tqdm import tqdm


def set_loader(opt):

    mod = opt.common_modality
    mod_space = ['acc', 'gyro', 'mag']
    multi_mod_space = [[mod, m] for m in mod_space if m != mod]

    opt.valid_mod = multi_mod_space

    if opt.dataset == "train_A":
        print(f"=\tTraining {opt.valid_mod[0]} on dataset A")
        opt.other_mod = opt.valid_mod[0][1]
        opt.missing_mod = opt.valid_mod[1][1]
        train_dataset = data.Multimodal_dataset([], opt.valid_mod[0], root='train_A', opt=opt)
    else:
        print(f"=\tTraining {opt.valid_mod[1]} on dataset B")
        opt.other_mod = opt.valid_mod[1][1]
        opt.missing_mod = opt.valid_mod[0][1]
        train_dataset = data.Multimodal_dataset([], opt.valid_mod[1], root='train_B', opt=opt)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)

    return train_loader


def set_model(opt):

    pprint(f"=\tInitializing Autoencoder for mod {opt.common_modality}")
    print(f"=\tInitializing Autoencoder for mod {opt.common_modality}")
    model = SingleIMUAutoencoder(opt.common_modality)  # acc autoencoder -> gyro/mag output

    model_weight = f"./save_baseline4/save_{opt.dataset}_cross_autoencoder_no_load_{opt.common_modality}_{opt.seed}_{opt.dataset_split}/models/lr_0.0001_decay_0.0001_bsz_64/last.pth"

    model.load_state_dict(torch.load(model_weight)['model'])

    # enable synchronized Batch Normalization
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    return model




def validate(val_loader, model, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()

    generated_data_list = []
    label_list = []
    original_data_list = []

    with torch.no_grad():
        end = time.time()
        for _, batched_data in tqdm(enumerate(val_loader)):

            labels = batched_data['action']
            if torch.cuda.is_available():
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            generated_data = model(batched_data)
            generated_data = torch.reshape(generated_data, (bsz, -1, 3))


            # calculate and store confusion matrix
            generated_data_list.extend(generated_data.cpu().numpy())
            original_data_list.extend(batched_data[opt.other_mod].cpu().numpy())
            label_list.extend(labels.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    generated_data_list = np.array(generated_data_list)
    original_data_list = np.array(original_data_list)
    label_list = np.array(label_list)

    return generated_data_list, original_data_list, label_list



def main():

    opt = parse_option("save_baseline4", "cross_generation")

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model = set_model(opt)

    generate_data_list, original_data_list, label_list = validate(train_loader, model, opt)

    save_paired_path = f"./save_baseline4/{opt.dataset}_generated_AB_{opt.common_modality}_{opt.seed}_{opt.dataset_split}/"
    if not os.path.isdir(save_paired_path):
        os.makedirs(save_paired_path + f"{opt.other_mod}/")
        os.makedirs(save_paired_path + f"{opt.missing_mod}/")

    for sample_index in range(len(label_list)):
        np.save(save_paired_path + f'{opt.other_mod}/{sample_index}.npy', original_data_list[sample_index])
        np.save(save_paired_path + f'{opt.missing_mod}/{sample_index}.npy', generate_data_list[sample_index])
        np.save(save_paired_path + 'label.npy', label_list)
    

if __name__ == '__main__':
    main()
