from __future__ import print_function

import os
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
# from torchvision import transforms, datasets


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from shared_files.util import AverageMeter
from shared_files.util import adjust_learning_rate, warmup_learning_rate, accuracy
from shared_files.util import set_optimizer, save_model
from shared_files import data_pre as data

from tqdm import tqdm
from modules.option_utils import parse_option
from modules.print_utils import pprint

from models.imu_models import SingleIMUAutoencoder



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
    print(f"=\tInitializing Autoencoder for mod {opt.other_mod}")
    model = SingleIMUAutoencoder(opt.other_mod) # Either gyro or mag

    # enable synchronized Batch Normalization
    # if opt.syncBN:
        # model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
            # model = torch.nn.DataParallel(model)
        model = model.cuda()
        # criterion = criterion.cuda()
        cudnn.benchmark = True

    return model


def train(train_loader, model, optimizer, epoch, opt):
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

        loss = F.mse_loss(batched_data[opt.other_mod].cuda(), output)

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

    opt = parse_option(exp_type="save_baseline1", exp_tag="autoencoder")

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    # tb_logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    record_loss = np.zeros(opt.epochs)

    # training routine
    pprint(f"Start Training")
    for epoch in tqdm(range(1, opt.epochs + 1), desc=f'Epoch: ', unit='items', ncols=80, colour='green', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining}]'):

        # Learning Rate
        adjust_learning_rate(opt, optimizer, epoch)

        # Forward pass
        loss = train(train_loader, model, optimizer, epoch, opt)

        # Update loss
        record_loss[epoch-1] = loss

        pprint(f"Epoch {epoch} - Loss: {loss}")
    
    # save the record loss
    np.savetxt(opt.result_path + f"loss_{opt.learning_rate}_{opt.epochs}.txt", record_loss)

    # save the last model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

if __name__ == '__main__':
    main()
