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
from models.imu_models import SingleIMUAutoencoder, GyroMagEncoder
from shared_files.contrastive_design import FeatureConstructor, ConFusionLoss

from tqdm import tqdm
from modules.option_utils import parse_option
from modules.print_utils import pprint

def set_loader(opt):

    if opt.dataset == "train_A":
        print("Training dataset A")
        train_dataset = data.Multimodal_dataset([], ['acc', 'gyro', 'mag'], root='../PAMAP_Dataset/trainA/')
    elif opt.dataset == 'train_B':
        print("Training dataset B")
        train_dataset = data.Multimodal_dataset([], ['acc', 'gyro', 'mag'], root='../PAMAP_Dataset/trainB/')
    elif opt.dataset == 'train_AB':
        print("Training dataset A and B")
        train_datasetA = data.Multimodal_dataset([], ['acc', 'gyro', 'mag'], root='../PAMAP_Dataset/trainA/')
        train_datasetB = data.Multimodal_dataset([], ['acc', 'gyro', 'mag'], root='../PAMAP_Dataset/trainB/')
        train_dataset = ConcatDataset([train_datasetA, train_datasetB])
    else:
        raise Exception("invalid dataset")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    return train_loader



def load_single_modal(opt, modality):

    if opt.load_pretrain == "load_pretrain":
        if modality == 'acc':
            opt.ckpt = './save_baseline1/save_train_A_autoencoder_no_load/models/lr_0.0001_decay_0.0001_bsz_64/'
        else:
            opt.ckpt = './save_baseline1/save_train_B_autoencoder_no_load/models/lr_0.0001_decay_0.0001_bsz_64/'
    elif opt.load_pretrain == "load_self_AE_pretrain":
        opt.ckpt = './save_upper_bound/unimodal_pretrain/save_{}_{}_autoencoder/models/lr_0.0001_decay_0.0001_bsz_64/'.format(opt.dataset, modality)

    ckpt_path = opt.ckpt + 'last.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    if modality == "acc":
        layer_key = 'acc_encoder.'
    else:
        layer_key = 'gyro_encoder.'
    new_state_dict = {}
    for k, v in state_dict.items():
        if layer_key in k:
            k = k.replace(layer_key, "")
            if torch.cuda.is_available():
                k = k.replace("module.", "")
            new_state_dict[k] = v
    state_dict = new_state_dict

    return state_dict

def set_model(opt):

    model = GyroMagEncoder()
    criterion = ConFusionLoss(temperature=opt.temp)

 
    if opt.load_pretrain == "load_pretrain":
        print("Loading pretrained weights from step 1")
        model_template = SingleIMUAutoencoder('acc')
        model_template.load_state_dict(torch.load('./save_upper_bound/unimodal_pretrain/save_train_AB_autoencoder_no_load_gyro/models/lr_0.0001_decay_0.0001_bsz_64/last.pth')['model'])
        model.gyro_encoder = model_template.encoder
        model_template.load_state_dict(torch.load('./save_upper_bound/unimodal_pretrain/save_train_AB_autoencoder_no_load_mag/models/lr_0.0001_decay_0.0001_bsz_64/last.pth')['model'])
        model.mag_encoder = model_template.encoder

       # enable synchronized Batch Normalization
    # if opt.syncBN:
        # model = apex.parallel.convert_syncbn_model(model)

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

        gyro_embed, mag_embed = model(batched_data)
        bsz = gyro_embed.shape[0]
        features = FeatureConstructor(gyro_embed, mag_embed, 2)

        loss = criterion(features)
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        # if (idx + 1) % opt.print_freq == 0:
        #     print('Train: [{0}][{1}/{2}]\t'
        #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
        #            epoch, idx + 1, len(train_loader), batch_time=batch_time,
        #            data_time=data_time, loss=losses))
        #     sys.stdout.flush()

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


        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
        
        pprint(f"Epoch {epoch} - Loss: {loss}")

    np.savetxt(opt.result_path + f"loss_{opt.learning_rate}_{opt.epochs}.txt", record_loss)
    
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)



if __name__ == '__main__':
    main()
