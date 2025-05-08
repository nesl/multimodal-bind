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

import yaml
import models.model as model_lib
from shared_files.PickleDataset import make_dataset
from torch.utils.data import ConcatDataset
import torch.optim as optim




try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''

    for i in range(len(batch)):
        if ('input_depth' not in batch[i].keys()):
            batch[i]['input_depth'] = torch.full((30, 1, 48, 64), -1.1)
        if ('input_mmwave' not in batch[i].keys()):
            batch[i]['input_mmwave'] = torch.full((1, 297, 5), -1.1)

    batch_data = {'modality': [batch[i]['modality'] for i in range(len(batch))],
                    'scene': [sample['scene'] for sample in batch],
                    'subject': [sample['subject'] for sample in batch],
                    'action': [sample['action'] for sample in batch],
                    'idx': [sample['idx'] for sample in batch] if 'idx' in batch[0] else None
                    }
    _output = [np.array(sample['output']) for sample in batch]
    _output = torch.FloatTensor(np.array(_output))
    batch_data['output'] = _output

    for mod in ['mmwave', 'depth']:
        if mod in ['mmwave', 'lidar']:
            _input = [torch.Tensor(sample['input_' + mod]) for sample in batch]
            _input = torch.nn.utils.rnn.pad_sequence(_input)
            _input = _input.permute(1, 2, 0, 3)
            batch_data['input_' + mod] = _input
        else:
            _input = [np.array(sample['input_' + mod]) for sample in batch]
            _input = torch.FloatTensor(np.array(_input))
            batch_data['input_' + mod] = _input
    return batch_data

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
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
    parser.add_argument('--load_pretrain', type=str, default='no_pretrain', help='load_pretrain')

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
    opt.save_path = "./save_baseline3/"
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

    # We load both trainA and trainB config file data
    if (opt.dataset == 'train_A'):
        with open('../Configs/config_train_A.yaml', 'r') as handle:
            config_train = yaml.load(handle, Loader=yaml.FullLoader)
        train_dataset, _ = make_dataset('/mnt/ssd_8t/jason/MMFI_Dataset/', config_train)

    elif (opt.dataset == 'train_B'):
        with open('../Configs/config_train_B.yaml', 'r') as handle:
            config_train = yaml.load(handle, Loader=yaml.FullLoader)
        train_dataset, _ = make_dataset('/mnt/ssd_8t/jason/MMFI_Dataset/', config_train)

    elif (opt.dataset == 'train_AB'):
        with open('../Configs/config_train_A.yaml', 'r') as handle:
            config_train = yaml.load(handle, Loader=yaml.FullLoader)
        train_datasetA, _ = make_dataset('/mnt/ssd_8t/jason/MMFI_Dataset/', config_train)
        with open('../Configs/config_train_B.yaml', 'r') as handle:
            config_train = yaml.load(handle, Loader=yaml.FullLoader)
        train_datasetB, _ = make_dataset('/mnt/ssd_8t/jason/MMFI_Dataset/', config_train)
        train_dataset = ConcatDataset([train_datasetA, train_datasetB])

  
    # Return single train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
         collate_fn=collate_fn_padd, pin_memory=True, shuffle=True)
    
    return train_loader


def set_model(opt):

    model = model_lib.IncompleteSupervisedPrompted()

    criterion = torch.nn.CrossEntropyLoss()

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
    top1 = AverageMeter()

    end = time.time()

    label_list = []
    pred1_list = []
    softmax = torch.nn.Softmax(dim=-1)
    for idx, batched_data in enumerate(train_loader):
        data_time.update(time.time() - end)

       
        bsz = len(batched_data['action'])

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        actions = batched_data['action']
        labels = torch.tensor([int(item[1:]) - 1 for item in actions]).cuda()
        mask = torch.zeros((bsz, 2))
        i = 0
        for valid_mods in batched_data['modality']:
            mask[i] = torch.tensor(['depth' in valid_mods, 'mmwave' in valid_mods])
            i += 1
        mask = mask.cuda()

        output = model(batched_data, mask)

        label_list.extend(labels.cpu().numpy())
        pred1_list.extend(output.max(1)[1].cpu().numpy())
        
        loss = criterion(output, labels)

        output = softmax(output)

        # acc, rec, prec, target_positive, pred_positive = rate_eval(output, labels, opt.num_class, topk=(1, 5))
        acc, _ = accuracy(output, labels, topk=(1, 5))
        # print("Compare acc, rec:", acc, rec)

        # update metric
        losses.update(loss.item(), bsz)
        top1.update(acc[0], bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    # print("label_list:",label_list, len(label_list))
    # print("pred1_list:",pred1_list)

    F1score = f1_score(label_list, pred1_list, average=None)
    # print('feature_f1:', F1score)


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



