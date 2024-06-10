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
from torch.utils.data import DataLoader, ConcatDataset

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from shared_files.util import AverageMeter
from shared_files.util import adjust_learning_rate, warmup_learning_rate, accuracy
from shared_files.util import set_optimizer, save_model
from shared_files import data_pre as data

from models.imu_models import FullIMUEncoder, SingleIMUAutoencoder
from shared_files.contrastive_design import FeatureConstructor, ConFusionLoss

from tqdm import tqdm
from modules.option_utils import parse_option
from modules.print_utils import pprint


def collate_fn_pad(batch):
    for i in range(len(batch)):
        if ('acc' not in batch[i]['valid_mods']):
            batch[i]['acc'] = np.zeros((1000, 3))
        if ('gyro' not in batch[i]['valid_mods']):
            batch[i]['gyro'] = np.zeros((1000, 3))
        if ('mag' not in batch[i]['valid_mods']):
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

    #load labeled train and test data
    mod = opt.common_modality
    mod_space = ['acc', 'gyro', 'mag']
    multi_mod_space = [[mod, m] for m in mod_space if m != mod]

    opt.valid_mod = multi_mod_space
    opt.other_mod = [opt.valid_mod[0][1], opt.valid_mod[1][1]]

    #load labeled train and test data
    if opt.dataset == "train_AB":
        print(f"=\tTraining with dataset AB with A - {opt.valid_mod[0]} and B - {opt.valid_mod[1]}")
        train_datasetA = data.Multimodal_dataset([], opt.valid_mod[0], root='train_A', opt=opt)
        train_datasetB = data.Multimodal_dataset([], opt.valid_mod[1],  root='train_B', opt=opt)
        train_dataset = ConcatDataset([train_datasetA, train_datasetB])
    elif opt.dataset == "train_A":
        print(f"=\tTraining with dataset A and valid mod {opt.valid_mod[0]}")
        train_dataset = data.Multimodal_dataset([], opt.valid_mod[0], root='train_A', opt=opt)
    elif opt.dataset == "train_B":
        print(f"=\tTraining with dataset B and valid mod {opt.valid_mod[1]}")
        train_dataset = data.Multimodal_dataset([], opt.valid_mod[1], root='train_B', opt=opt)
    else:
        raise Exception("Invalid dataset selection")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, collate_fn = collate_fn_pad, num_workers=opt.num_workers,
         pin_memory=True, shuffle=True, drop_last=True)
    return train_loader



def load_single_modal(opt, modality):

    if opt.load_pretrain == "load_pretrain":
        if modality == 'gyro':
            opt.ckpt = './save_baseline1/save_train_A_autoencoder/models/lr_0.005_decay_0.0001_bsz_64/'
        else:
            opt.ckpt = './save_baseline1/save_train_B_autoencoder/models/lr_0.005_decay_0.0001_bsz_64/'

    ckpt_path = opt.ckpt + 'last.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    # if modality == "acc":
    #     layer_key = 'acc_encoder.'
    # else:
    #     layer_key = 'gyro_encoder.'
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     if layer_key in k:
    #         k = k.replace(layer_key, "")
    #         if torch.cuda.is_available():
    #             k = k.replace("module.", "")
    #         new_state_dict[k] = v
    # state_dict = new_state_dict

    return state_dict



def set_model(opt):

    model = FullIMUEncoder()
    criterion = ConFusionLoss(temperature=opt.temp)

    if opt.load_pretrain == "load_pretrain":
        model_template = SingleIMUAutoencoder('gyro')
        print(f"=\tLoading pretrain model")
        model_template.load_state_dict(load_single_modal(opt, 'gyro'))
        model.gyro_encoder = model_template.encoder
        model_template.load_state_dict(load_single_modal(opt, 'mag'))
        model.acc_encoder = model_template.encoder
    
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


        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        
        # acc is the shared feature
        acc_embed, gyro_embed, mag_embed = model(batched_data)
        mod_embed = {
            "acc": acc_embed,
            "gyro": gyro_embed,
            "mag": mag_embed
        }
        
        bsz = acc_embed.shape[0]

        common_mod_embed = mod_embed[opt.common_modality]


        """
        valid_mods = batched_data['valid_mods']

        gyro_mask = torch.tensor([1 if 'gyro' in mods else 0 for mods in valid_mods])
        mag_mask = torch.tensor([1 if 'mag' in mods else 0 for mods in valid_mods])
        gyro_mask = torch.unsqueeze(gyro_mask, dim=-1).cuda()
        mag_mask = torch.unsqueeze(mag_mask, dim=-1).cuda()

        combined_features = gyro_mask * gyro_embed + mag_mask * mag_embed
        features = FeatureConstructor(acc_embed, combined_features, 2)
        """



        valid_mods = batched_data['valid_mods'] # [common mod, mod1] or [common mod, mod2]

        other_mod1, other_mod2 = opt.other_mod

        mod1_embed = mod_embed[other_mod1]
        mod2_embed = mod_embed[other_mod2]

        mod1_mask = torch.tensor([1 if other_mod1 in mods else 0 for mods in valid_mods])
        mod2_mask = torch.tensor([1 if other_mod2 in mods else 0 for mods in valid_mods])

        mod1_mask = torch.unsqueeze(mod1_mask, dim=-1).cuda()
        mod2_mask = torch.unsqueeze(mod2_mask, dim=-1).cuda()

        combined_features = mod1_mask * mod1_embed + mod2_mask * mod2_embed

        features = FeatureConstructor(common_mod_embed, combined_features, 2)

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
    opt = parse_option("save_dual_contrastive", "contrastive")

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


        # if epoch % opt.save_freq == 0:
        #     save_file = os.path.join(
        #         opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     save_model(model, optimizer, opt, epoch, save_file)

        np.savetxt(opt.result_path + "loss_{}_{}.txt".format(opt.learning_rate, opt.batch_size), record_loss)
    
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)



if __name__ == '__main__':
    main()
