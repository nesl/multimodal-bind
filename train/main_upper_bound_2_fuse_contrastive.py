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
from torch.utils.data import ConcatDataset
from models.imu_models import ModEncoder, SingleIMUAutoencoder, GyroMagEncoder
from shared_files.contrastive_design import FeatureConstructor, ConFusionLoss

from tqdm import tqdm
from modules.option_utils import parse_option
from modules.print_utils import pprint

def set_loader(opt):

    if opt.dataset == "train_A":
        print("Training dataset A")
        train_dataset = data.Multimodal_dataset([], ['acc', 'gyro', 'mag'], root='train_A', opt=opt)
    elif opt.dataset == 'train_B':
        print("Training dataset B")
        train_dataset = data.Multimodal_dataset([], ['acc', 'gyro', 'mag'], root='train_B', opt=opt)
    elif opt.dataset == 'train_AB':
        print("Training dataset A and B")
        train_datasetA = data.Multimodal_dataset([], ['acc', 'gyro', 'mag'], root='train_A', opt=opt)
        train_datasetB = data.Multimodal_dataset([], ['acc', 'gyro', 'mag'], root='train_B', opt=opt)
        train_dataset = ConcatDataset([train_datasetA, train_datasetB])
    else:
        raise Exception("invalid dataset")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    return train_loader


def set_model(opt):

    mod = opt.common_modality
    mod_space = ['acc', 'gyro', 'mag']
    multi_mod_space = [[mod, m] for m in mod_space if m != mod]

    opt.valid_mod = multi_mod_space

    model = ModEncoder()
    criterion = ConFusionLoss(temperature=opt.temp)

 
    if opt.load_pretrain == "load_pretrain":
        print("=\tLoading pretrained weights from step 1")
        print(f"=\tLoading IMUEncoder for {mod}")
        model_template = SingleIMUAutoencoder(mod) # any mod should be fine

        mod1 = opt.valid_mod[0][1]
        mod2 = opt.valid_mod[1][1]
        model_template.load_state_dict(torch.load(f'./save_upper_bound/unimodal_pretrain/save_train_AB_autoencoder_no_load_{mod1}/models/lr_0.0001_decay_0.0001_bsz_64/last.pth')['model'])
        setattr(model, f"{mod1}_encoder", model_template.encoder)

        model_template.load_state_dict(torch.load(f'./save_upper_bound/unimodal_pretrain/save_train_AB_autoencoder_no_load_{mod2}/models/lr_0.0001_decay_0.0001_bsz_64/last.pth')['model'])
        setattr(model, f"{mod2}_encoder", model_template.encoder)

    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
            # model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True


    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    for idx, batched_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        acc_embed, gyro_embed, mag_embed = model(batched_data)
        embed = {
            "acc": acc_embed,
            "gyro": gyro_embed,
            "mag": mag_embed
        }

        bsz = gyro_embed.shape[0]

        embed1 = embed[opt.valid_mod[0][1]]
        embed2 = embed[opt.valid_mod[1][1]]
        features = FeatureConstructor(embed1, embed2, 2)

        loss = criterion(features)
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


def main():
    opt = parse_option("save_upper_bound", "fuse_contrastive")

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    record_loss = np.zeros(opt.epochs)

    # training routine
    pprint(f"Start Training")
    for epoch in tqdm(range(1, opt.epochs + 1), desc=f'Epoch: ', unit='items', ncols=80, colour='green', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining}]'):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)

        record_loss[epoch-1] = loss
        
        pprint(f"Epoch {epoch} - Loss: {loss}")

    np.savetxt(opt.result_path + f"loss_{opt.learning_rate}_{opt.epochs}.txt", record_loss)
    
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)



if __name__ == '__main__':
    main()
