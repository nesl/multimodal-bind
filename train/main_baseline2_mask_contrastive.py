from __future__ import print_function

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
from shared_files.util import adjust_learning_rate, warmup_learning_rate, accuracy
from shared_files.util import set_optimizer, save_model
from shared_files import data_pre as data

from models.imu_models import FullIMUEncoder, SingleIMUAutoencoder
from shared_files.contrastive_design_3M import FeatureConstructor, ConFusionLoss



try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def collate_fn_pad(batch):
    for i in range(len(batch)):
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

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')

    # model dataset
    parser.add_argument('--model', type=str, default='MyUTDmodel')
    parser.add_argument('--dataset', type=str, default='train_AB',
                        choices=['train_A', 'train_B', 'train_AB'], help='dataset')
    parser.add_argument('--num_class', type=int, default=27,
                        help='num_class')
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--load_pretrain', type=str, default='no_load', help='load_pretrain')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--seed', type=int, default=100)


    opt = parser.parse_args()

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # set the path according to the environment
    opt.save_path = "./save_baseline2/save_{}_mask_contrastive_{}/".format(opt.dataset, opt.load_pretrain)
    opt.model_path = opt.save_path + 'models'
    opt.tb_path = opt.save_path + 'tensorboard'
    opt.result_path = opt.save_path + 'results/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'lr_{}_decay_{}_bsz_{}'.\
        format(opt.learning_rate, opt.weight_decay, opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)

    return opt


def set_loader(opt):

    #load labeled train and test data
    print("train data:")
    if opt.dataset == "train_AB":
        print("Training with dataset AB")
        train_datasetA = data.Multimodal_dataset([], ['acc', 'gyro'], root='../../PAMAP_Dataset/trainA/')
        train_datasetB = data.Multimodal_dataset([], ['acc', 'mag'],  root='../../PAMAP_Dataset/trainB/')
        train_dataset = ConcatDataset([train_datasetA, train_datasetB])
    elif opt.dataset == "train_A":
        print("Training with dataset A only. Are you sure??")
        train_dataset = data.Multimodal_dataset([], ['acc', 'gyro'], root='../../PAMAP_Dataset/trainA/')
    elif opt.dataset == "train_B":
        print("Training with dataset B only. Are you sure??")
        train_dataset = data.Multimodal_dataset([], ['acc', 'gyro'], root='../../PAMAP_Dataset/trainB/')
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

    return state_dict


def set_model(opt):

    model = FullIMUEncoder()
    model_template = SingleIMUAutoencoder('gyro')
    criterion = ConFusionLoss(temperature=opt.temp)

    if opt.load_pretrain == "load_pretrain":
        model_template.load_state_dict(load_single_modal(opt, 'gyro'))
        model.gyro_encoder = model_template.encoder
        model_template.load_state_dict(load_single_modal(opt, 'mag'))
        model.acc_encoder = model_template.encoder
    
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

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

    for idx, batched_data in enumerate(train_loader):

        data_time.update(time.time() - end)


        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        acc_embed, gyro_embed, mag_embed = model(batched_data)
        bsz = acc_embed.shape[0]

        features = FeatureConstructor(acc_embed, gyro_embed, mag_embed, 3)

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
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

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
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()

        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        record_loss[epoch-1] = loss


        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        np.savetxt(opt.result_path + "loss_{}_{}.txt".format(opt.learning_rate, opt.batch_size), record_loss)
    
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)



if __name__ == '__main__':
    main()
