from __future__ import print_function

import os
import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
# from torchvision import transforms, datasets


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from shared_files.util import AverageMeter
from shared_files.util import adjust_learning_rate, warmup_learning_rate, accuracy
from shared_files.util import set_optimizer, save_model
from shared_files import data_pre as data

from models.imu_models import SingleIMUAutoencoder
from tqdm import tqdm
from modules.option_utils import parse_option
from modules.print_utils import pprint


def set_loader(opt):
    if opt.dataset == "train_A":
        print("Training dataset A")
        train_dataset = data.Multimodal_dataset([], ['acc'], root='train_A', opt=opt)
    elif opt.dataset == 'train_B':
        print("Training dataset B")
        train_dataset = data.Multimodal_dataset([], ['acc'], root='train_B', opt=opt)
    elif opt.dataset == 'train_AB':
        print("Training dataset A and B")
        train_datasetA = data.Multimodal_dataset([], ['acc'], root='train_A', opt=opt)
        train_datasetB = data.Multimodal_dataset([], ['acc'], root='train_B', opt=opt)
        train_dataset = ConcatDataset([train_datasetA, train_datasetB])
    else:
        raise Exception("invalid dataset")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)

    return train_loader


def set_model(opt):

    mod = opt.common_modality
    print(f"=\tLoading Autoencoder for modality {mod}")
    model = SingleIMUAutoencoder(mod) 

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


    for idx, batched_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        labels = batched_data['action']
        if torch.cuda.is_available():
            labels = labels.cuda()
        bsz = labels.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        output = model(batched_data)
        output = torch.reshape(output, (bsz, -1, 3))
        loss = F.mse_loss(batched_data['acc'].cuda(), output)

        # update metric
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

    # print(output[0][0])
    return losses.avg


def main():

    opt = parse_option("save_imagebind", "acc_autoencoder")

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    record_loss = np.zeros(opt.epochs)

    pprint(f"Start Training")
    for epoch in tqdm(range(1, opt.epochs + 1), desc=f'Epoch: ', unit='items', ncols=80, colour='green', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining}]'):
        adjust_learning_rate(opt, optimizer, epoch)

        loss = train(train_loader, model, optimizer, epoch, opt)

        record_loss[epoch-1] = loss

        # if epoch % opt.save_freq == 0:
        #     save_file = os.path.join(
        #         opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     save_model(model, optimizer, opt, epoch, save_file)

        pprint(f"Epoch {epoch} - Loss: {loss}")

    np.savetxt(opt.result_path + f"loss_{opt.learning_rate}_{opt.epochs}.txt", record_loss)
    
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)




if __name__ == '__main__':
    main()
