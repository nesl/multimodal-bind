import os
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn

import numpy as np

from shared_files.util import AverageMeter
from shared_files.util import adjust_learning_rate
from shared_files.util import set_optimizer, save_model
from shared_files import data_pre as data

from models.imu_models import GyroMagEncoder, SingleIMUAutoencoder, ModEncoder, FullIMUEncoder
from shared_files.contrastive_design_3M import FeatureConstructor, ConFusionLoss

from tqdm import tqdm
from modules.option_utils import parse_option
from modules.print_utils import pprint

def collate_fn_pad(batch):
    for i in range(len(batch)):
        if ('acc' not in batch[i]['valid_mods']):
            batch[i]['acc'] = np.zeros((1000, 3))
        if ('gyro' not in batch[i]['valid_mods']):
            print(f"Masking gyro!")
            batch[i]['gyro'] = np.zeros((1000, 3))
        if ('mag' not in batch[i]['valid_mods']):
            print(f"Masking mag!")
            batch[i]['mag'] = np.zeros((1000, 3))
    batched_data = {
        'valid_mods': [sample['valid_mods'] for sample in batch],
        'gyro': [torch.FloatTensor(sample['gyro']) for sample in batch],
        'acc': [torch.FloatTensor(sample['acc']) for sample in batch],
        'mag': [torch.FloatTensor(sample['mag']) for sample in batch],
    }
    batched_data['gyro'] = torch.stack(batched_data['gyro'])
    batched_data['mag'] = torch.stack(batched_data['mag'])
    batched_data['acc'] = torch.stack(batched_data['acc'])
    return batched_data

def set_loader(opt):
    print(f"=\tInitializing Dataloader")

    dataset = f"./save_mmbind/train_all_paired_AB_{opt.common_modality}_{opt.seed}_{opt.dataset_split}/"

    pprint(f"=\tLoading dataset from {dataset}")
    print(f"=\tLoading dataset from {dataset}")

    mod = opt.common_modality
    mod_space = ['acc', 'gyro', 'mag']
    multi_mod_space = [[mod, m] for m in mod_space if m != mod]

    opt.valid_mod = multi_mod_space
    opt.non_common_mod = [m for m in mod_space if m != mod]


    pprint(f"=\tLoading dataset from {dataset}")
    print(f"=\tLoading dataset from {dataset}")
    train_dataset = data.Multimodal_dataset([], opt.non_common_mod, root=dataset, opt=opt)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, collate_fn=collate_fn_pad, num_workers=opt.num_workers, pin_memory=True, shuffle=True, drop_last=True)

    return train_loader

def set_model(opt):

    model = FullIMUEncoder()
    criterion = ConFusionLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization

    if torch.cuda.is_available():
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

    common_modality = opt.common_modality
    other_modalities = [m for m in ['acc', 'gyro', 'mag'] if m != common_modality]

    for idx, batched_data in enumerate(train_loader):
        data_time.update(time.time() - end)


        acc_embed, gyro_embed, mag_embed = model(batched_data)
        bsz = acc_embed.shape[0]

        features = torch.stack([acc_embed, gyro_embed, mag_embed], dim=1)

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
    opt = parse_option("save_mmbind", "incomplete_contrastive")

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
        pprint(f"Epoch: {epoch} - Loss: {loss}")
    
    np.savetxt(opt.result_path + f"loss_{opt.learning_rate}_{opt.epochs}.txt", record_loss)
    
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)



if __name__ == '__main__':
    main()
