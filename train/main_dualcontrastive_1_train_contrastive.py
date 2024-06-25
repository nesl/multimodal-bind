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

from shared_files.contrastive_design import FeatureConstructor, ConFusionLoss
import yaml
from models.model import DualContrastiveModel
from shared_files.PickleDataset import make_dataset
from torch.utils.data import ConcatDataset

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

# Dual Contrastive


# Dual contrastive collate_fn_padd is a bit different
# We place dummy data for the unpaired mmWave and depth samples
# This dummy data goes into the model and becomes an embedding, but we mask them out later in train
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

    for mod in ['mmwave', 'depth', 'rgb']:
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
    parser.add_argument('--epochs', type=int, default=100,
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
    parser.add_argument("--train_config", type=str, default="train_AB", help="Configuration YAML file")
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
    opt.save_path = "./save_dual_contrastive/save_{}_contrastive_{}/".format(opt.train_config, opt.load_pretrain)
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
    with open('../Configs/config_train_A.yaml', 'r') as handle:
        config_train = yaml.load(handle, Loader=yaml.FullLoader)

    train_datasetA, _ = make_dataset('../../MMFI_Dataset/', config_train)

    with open('../Configs/config_train_B.yaml', 'r') as handle:
        config_train = yaml.load(handle, Loader=yaml.FullLoader)

    train_datasetB, _ = make_dataset('../../MMFI_Dataset/', config_train)

    # Place them into a single dataset! This is why we had to generate dummy data samples to ensure consistent shape within batch
    train_dataset = ConcatDataset([train_datasetA, train_datasetB])

    # Return single train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, num_workers = opt.num_workers,
         collate_fn=collate_fn_padd, pin_memory=True, shuffle=True)
    

    return train_loader

# We set the model here, note that we are not loading any pretrained weights
def set_model(opt):

    model = DualContrastiveModel()
    model = model.cuda()
    criterion = ConFusionLoss(temperature=opt.temp)

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

        bsz = batched_data['input_rgb'].shape[0]

        # Model returns three adapted embeddings (depth, mmwave, skeleton), input is entire dict
        depth_features, mmwave_features, skeleton_features = model(batched_data)

        # Within valid_mods, we have b_size number of lists ['mmwave', 'rgb'] or ['depth', 'rgb']
        valid_mods = batched_data['modality']
        valid_mods = np.array(valid_mods)
        # Generate mask, if entry in batch has the modality it is 1, else 0. Dummy data is given 0, real data 1
        mmwave_mask = torch.tensor([1 if 'mmwave' in valids else 0 for valids in valid_mods]).cuda()
        depth_mask = torch.tensor([1 - mmwave_val for mmwave_val in mmwave_mask]).cuda()

        # Mask out invalid entries and combine
        mmwave_mask = torch.unsqueeze(mmwave_mask, -1)
        depth_mask = torch.unsqueeze(depth_mask, -1)
        # Combined features now holds either depth_features or mmwave features 
        combined_features = depth_features * depth_mask + mmwave_mask * mmwave_features
        
        # Perform contrastive learning
        features = FeatureConstructor(skeleton_features, combined_features, 2)
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
