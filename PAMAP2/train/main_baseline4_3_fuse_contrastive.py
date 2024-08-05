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

from torch.utils.data import ConcatDataset
from shared_files.util import AverageMeter, adjust_learning_rate, set_optimizer, save_model
from shared_files import data_pre as data

from models.imu_models import FullIMUEncoder, SingleIMUAutoencoder
from shared_files.contrastive_design_3M import ConFusionLoss
from shared_files.contrastive_design import FeatureConstructor

from tqdm import tqdm
from modules.option_utils import parse_option
from modules.print_utils import pprint

def collate_fn_pad(batch):
    for i in range(len(batch)):
        if ('acc' not in batch[i]):
            batch[i]['acc'] = np.zeros((1000, 3))
        if ('gyro' not in batch[i]):
            batch[i]['gyro'] = np.zeros((1000, 3))
        if ('mag' not in batch[i]):
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

    mod = opt.common_modality
    mod_space = ['acc', 'gyro', 'mag']
    multi_mod_space = [[mod, m] for m in mod_space if m != mod]

    opt.valid_mod = multi_mod_space

    print(f"=\tInitializing Dataloader")
    #load labeled train and test data

    dataset_A = f"./save_baseline4/train_A_generated_AB_{opt.common_modality}_{opt.seed}_{opt.dataset_split}/"
    dataset_B = f"./save_baseline4/train_B_generated_AB_{opt.common_modality}_{opt.seed}_{opt.dataset_split}/"
    
    pprint(f"=\tLoading dataset from {dataset_A}")
    print(f"=\tLoading dataset from {dataset_A}")

    pprint(f"=\tLoading dataset from {dataset_B}")
    print(f"=\tLoading dataset from {dataset_B}")

    train_datasetA = data.Multimodal_generated_dataset([], opt.valid_mod[0], root=dataset_A, opt=opt)
    train_datasetB = data.Multimodal_generated_dataset([], opt.valid_mod[1], root=dataset_B, opt=opt)
    train_dataset = ConcatDataset([train_datasetA, train_datasetB])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, collate_fn=collate_fn_pad, 
        num_workers=opt.num_workers, pin_memory=True, shuffle=True, drop_last=True)

    return train_loader



def load_single_modal(opt, layer_key):

    if layer_key == 'acc_encoder.':
        opt.ckpt = './save_baseline1/save_train_A_autoencoder/models/lr_0.0001_decay_0.0001_bsz_64/'
    elif layer_key == 'gyro_encoder.':
        opt.ckpt = './save_baseline1/save_train_B_autoencoder/models/lr_0.0001_decay_0.0001_bsz_64/'

    ckpt_path = opt.ckpt + 'last.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

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

    model = FullIMUEncoder()
    criterion = ConFusionLoss(temperature=opt.temp)

    if opt.load_pretrain == "load_pretrain":
        pprint(f"Loading pretrained model")
        print(f"=\tLoading pretrained model")

        mod1, mod2 = opt.valid_mod[0][1], opt.valid_mod[1][1]

        mod1_weight = f"./save_baseline1/save_train_A_autoencoder_no_load_{opt.common_modality}_{opt.seed}_{opt.dataset_split}/models/lr_0.0001_decay_0.0001_bsz_64/last.pth"
        mod2_weight = f"./save_baseline1/save_train_A_autoencoder_no_load_{opt.common_modality}_{opt.seed}_{opt.dataset_split}/models/lr_0.0001_decay_0.0001_bsz_64/last.pth"

        model_template = SingleIMUAutoencoder('acc')
        state_dict_1 = torch.load(mod1_weight)['model']
        model_template.load_state_dict(state_dict_1)
        setattr(model, f"{mod1}_encoder", model_template.encoder)

        state_dict_2 = torch.load(mod2_weight)['model']
        model_template.load_state_dict(state_dict_2)
        setattr(model, f"{mod2}_encoder", model_template.encoder)
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
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
    opt = parse_option("save_baseline4", "fuse_contrastive")

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

    
    # save the last model
    np.savetxt(opt.result_path + f"loss_{opt.learning_rate}_{opt.epochs}.txt", record_loss)
    
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)



if __name__ == '__main__':
    main()
