from __future__ import print_function

import os
import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from torchvision import transforms, datasets

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from shared_files.util import AverageMeter
from shared_files.util import adjust_learning_rate, warmup_learning_rate, accuracy
from shared_files.util import set_optimizer, save_model
import models.model as model_lib
from shared_files.PickleDataset import make_dataset


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

# By default we are training depth model first!!

# Pad batch appropriately for variable length data
def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''

    batch_data = {'modality': batch[0]['modality'],
                  'scene': [sample['scene'] for sample in batch],
                  'subject': [sample['subject'] for sample in batch],
                  'action': [sample['action'] for sample in batch],
                  'idx': [sample['idx'] for sample in batch] if 'idx' in batch[0] else None
                  }
    _output = [np.array(sample['output']) for sample in batch]
    _output = torch.FloatTensor(np.array(_output))
    batch_data['output'] = _output

    for mod in batch_data['modality']:
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
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=3e-4, # 3e-4 worked on combined multimodal model. Doesnt work 1e-4 and 1e-3
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
    # TODO changed
    parser.add_argument("--dataset", type=str, default="train_A", help="Configuration YAML file") # Provide either train_A or train_B
    parser.add_argument('--num_class', type=int, default=27,
                        help='num_class')


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
    opt.save_path = './save_baseline1/save_{}_autoencoder/'.format(opt.dataset)
    opt.model_path = opt.save_path + 'models'
    opt.tb_path = opt.save_path + 'tensorboard'
    opt.result_path = opt.save_path + 'results/'
    # Convert the input dataset into the config file name
    opt.train_path = '../Configs/config_' + opt.dataset + '.yaml'

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

# TODO changed
def set_loader(opt):

    # Load the data from config file. Note that this contains labels but we just don't use them
    with open(opt.train_path, 'r') as handle:
        config_train = yaml.load(handle, Loader=yaml.FullLoader)

    train_dataset, _ = make_dataset('../../MMFI_Dataset/', config_train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
         collate_fn=collate_fn_padd, pin_memory=True, shuffle=True)


    return train_loader



# TODO changed
def set_model(opt):
    # Create appropriate model depending on what we're trainng
    # Dataset A is for depth, dataset B is for mmWave
    if (opt.dataset == 'train_A'):
        print("Currently training depth model in dataset A")
        model_encoder = model_lib.DepthEncoder()
        model_decoder = model_lib.DepthReconstruct()
    elif (opt.dataset == 'train_B'):
        print("Curently training mmWave on dataset B")
        model_encoder = model_lib.mmWaveEncoder()
        model_decoder = model_lib.mmWaveReconstruct()
    else:
        raise Exception('Invalid dataset selected')
    
    criterion = nn.MSELoss()


    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model_encoder = torch.nn.DataParallel(model_encoder)
            model_decoder = torch.nn.DataParallel(model_decoder)
            
        model_encoder = model_encoder.cuda()
        model_decoder = model_decoder.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True


    return model_encoder, model_decoder, criterion


def train(train_loader, model_encoder, model_decoder, criterion, optimizer, epoch, opt):
    """one epoch training"""
    curr_mod = 'input_depth' if opt.dataset == 'train_A' else 'input_mmwave'
    model_encoder.train()
    model_decoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()


    end = time.time()

    label_list = []
    pred1_list = []
    for idx, batched_data in enumerate(train_loader):
        data_time.update(time.time() - end)


        bsz = len(batched_data['action'])
        # Encoder takes the batched_data, reconstructed takes the embedding and remakes it
        embedding = model_encoder(batched_data)
        reconstructed = model_decoder(embedding)
        
        # mmWave is interesting bc we have variable number of points
        if (curr_mod == 'input_mmwave'):
            # Get 30 frames of data to compare against during reconstruction
            raw_data = batched_data['input_mmwave'][:, 0:30].cuda()
            
            num_points = raw_data.shape[2] # batch_size, 30, ? , 5
            # If we have more than 100 points per frame ignore the rest
            if (num_points > 100):
                num_points = 100
                raw_data = raw_data[:, :, :100]
            # Reshape the reconstructed data. The reconstructed data makes 100 points of 5, get the number of values we need
            reconstructed = torch.reshape(reconstructed[:, :, 0:num_points * 5], (bsz, 30, num_points, -1))
            # collate_fn_pad will pad with zeros, zeros will indicate padded data
            mask = torch.where(raw_data != 0, 0, 1)
            # Set the parts where raw_data = 0 to the reconstructed so error will be zero
            raw_data = raw_data + mask * reconstructed # ignore the zero points for loss
        # I tried to do a difference between frames to get rid of background information, but didn't help much
        # Can add appropriate processing algorithms here to try to improve depth autoencoder performance
        elif (curr_mod == 'input_depth'):
            raw_data = batched_data['input_depth'][:, 0:30].cuda()

        # Calculate loss and backprop, everything else after this standard
        loss = criterion(torch.squeeze(reconstructed), torch.squeeze(raw_data))

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
    best_acc = 0
    best_rec = 0
    best_prec = 0
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model_encoder, model_decoder, criterion = set_model(opt)

    # build optimizer
    # optimizer = set_optimizer(opt, model)

    # build optimizer, add both the encoder and decoder parameters for gradient descent!
    optimizer = optim.Adam(list(model_encoder.parameters()) + list(model_decoder.parameters()), lr=opt.learning_rate,
                weight_decay=opt.weight_decay)

    # tensorboard
    #logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    record_loss = np.zeros(opt.epochs)


    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model_encoder, model_decoder, criterion, optimizer, epoch, opt)
        time2 = time.time()

        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))


        # evaluation
     

   
        
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model_encoder, optimizer, opt, epoch, save_file)

 
    
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model_encoder, optimizer, opt, opt.epochs, save_file)





if __name__ == '__main__':
    main()
