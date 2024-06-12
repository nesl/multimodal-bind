from __future__ import print_function

import os
import pprint
import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
from modules.option_utils import parse_option
from models.imu_models import SingleIMUAutoencoder
import torch
import torch.backends.cudnn as cudnn
# from torchvision import transforms, datasets


import numpy as np

from shared_files.util import AverageMeter
from shared_files.util import adjust_learning_rate
from shared_files.util import set_optimizer, save_model
from shared_files import data_pre as data

from tqdm import tqdm

def set_loader(opt):
    mod = opt.common_modality
    mod_space = ['acc', 'gyro', 'mag']
    multi_mod_space = [[mod, m] for m in mod_space if m != mod]

    opt.valid_mod = multi_mod_space

    if opt.dataset == "train_A":
        print(f"=\tTraining {opt.valid_mod[0]} on dataset A")
        opt.other_mod = opt.valid_mod[0][1]
        train_dataset = data.Multimodal_dataset([], opt.valid_mod[0], root='train_A', opt=opt)
    else:
        print(f"=\tTraining {opt.valid_mod[1]} on dataset B")
        opt.other_mod = opt.valid_mod[1][1]
        train_dataset = data.Multimodal_dataset([], opt.valid_mod[1], root='train_B', opt=opt)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)

    return train_loader


def set_model(opt):    
    pprint(f"=\tInitializing Autoencoder for mod {opt.common_modality}")
    print(f"=\tInitializing Autoencoder for mod {opt.common_modality}")
    model = SingleIMUAutoencoder(opt.common_modality)  # acc autoencoder -> gyro/mag output


    criterion = torch.nn.MSELoss()

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
    top1 = AverageMeter()

    end = time.time()


    for _, batched_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        labels = batched_data['action']
        if torch.cuda.is_available():
            labels = labels.cuda()
        bsz = labels.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        output = model(batched_data)
        output = torch.reshape(output, (bsz, -1, 3))

        loss = criterion(batched_data[opt.other_mod], output)

        # update metric
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
    opt = parse_option(exp_type="save_baseline4", exp_tag="cross_autoencoder")

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    record_loss = np.zeros(opt.epochs)

    pprint(f"Start Training")
    # training routine
    for epoch in tqdm(range(1, opt.epochs + 1), desc=f'Epoch: ', unit='items', ncols=80, colour='green', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining}]'):
        adjust_learning_rate(opt, optimizer, epoch)

        loss = train(train_loader, model, criterion, optimizer, epoch, opt)

        record_loss[epoch-1] = loss

        pprint(f"Epoch {epoch} - Loss: {loss}")
    
    # save the record loss
    np.savetxt(opt.result_path + f"loss_{opt.learning_rate}_{opt.epochs}.txt", record_loss)

    # save the last model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)




if __name__ == '__main__':
    main()
