import os
import sys
import argparse

import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset

# from torchvision import transforms, datasets

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


from shared_files.util import AverageMeter
from shared_files.util import adjust_learning_rate
from shared_files.util import set_optimizer, save_model
from shared_files import data_pre as data

from shared_files.contrastive_design_3M import ConFusionLoss

from tqdm import tqdm
from modules.option_utils import parse_option
from modules.print_utils import pprint
from models.imu_models import MaskedModEncoder


def set_loader(opt):

    mod = opt.common_modality
    mod_space = ["acc", "gyro", "mag"]

    multi_mod_space = [[mod, m] for m in mod_space if m != mod]
    opt.valid_mod = multi_mod_space

    print(f"=\tTraining with dataset AB with A - {opt.valid_mod[0]} and B - {opt.valid_mod[1]}")
    train_datasetA = data.Multimodal_masked_dataset([], opt.valid_mod[0], root="train_A", opt=opt)
    train_datasetB = data.Multimodal_masked_dataset([], opt.valid_mod[1], root="train_B", opt=opt)
    train_dataset = ConcatDataset([train_datasetA, train_datasetB])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    return train_loader


def set_model(opt):


    model = MaskedModEncoder()
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

    for _, batched_data in enumerate(train_loader):

        acc_embed, gyro_embed, mag_embed = model(batched_data)

        features = torch.stack([acc_embed, gyro_embed, mag_embed], dim=1)

        loss = criterion(features)
        losses.update(loss.item(), features.shape[0])

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


def main():
    opt = parse_option(exp_type="save_baseline3", exp_tag="vector2_incomplete_contrastive_zero")

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    record_loss = np.zeros(opt.epochs)

    # training routine
    pprint(f"Start Training")
    for epoch in tqdm(
        range(1, opt.epochs + 1),
        desc=f"Epoch: ",
        unit="items",
        ncols=80,
        colour="green",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining}]",
    ):
        adjust_learning_rate(opt, optimizer, epoch)

        loss = train(train_loader, model, criterion, optimizer, epoch, opt)

        record_loss[epoch - 1] = loss

        pprint(f"Epoch {epoch} - Loss: {loss}")

    # save the last model
    np.savetxt(opt.result_path + f"loss_{opt.learning_rate}_{opt.epochs}.txt", record_loss)
    save_file = os.path.join(opt.save_folder, "last.pth")
    save_model(model, optimizer, opt, opt.epochs, save_file)



if __name__ == "__main__":
    main()