from __future__ import print_function

import os
import sys
import argparse
import time
import math
import random

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from torchvision import transforms, datasets

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from shared_files.util import AverageMeter
from shared_files.util import adjust_learning_rate, warmup_learning_rate, accuracy
from shared_files.util import set_optimizer, save_model
from shared_files import data_pre as data

from models.imu_models import DualSupervisedIMUEncoder

from tqdm import tqdm
from modules.option_utils import parse_evaluation_option
from modules.print_utils import pprint

def load_dual_modal_set(opt, mod, root):

    print(f"=\tLoading data {mod} from {root}")
    train_dataset = data.Multimodal_dataset([], mod, root=root, opt=opt)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    
    x1 = []
    x2 = []
    y = []
    for _, batch in enumerate(train_loader):
        x1_data = batch[mod[0]]
        x2_data = batch[mod[1]]
        labels = batch['action']

        x1.append(x1_data)
        x2.append(x2_data)
        y.append(labels)
    
    # [nb] -> [nb * b]
    x1 = torch.concatenate(x1, dim=0)
    x2 = torch.concatenate(x2, dim=0)
    y = torch.concatenate(y, dim=0)

    return x1, x2, y


def set_loader(opt):
    x1_train, x2_train, y_train = load_dual_modal_set(opt, [opt.mod1, opt.mod2], "train_C")
    x1_test, x2_test, y_test = load_dual_modal_set(opt, [opt.mod1, opt.mod2], "test")

    train_dataset = data.Multimodal_dataset_direct_load(x1_train, x2_train, y_train)
    test_dataset = data.Multimodal_dataset_direct_load(x1_test, x2_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)

    return train_loader, val_loader


def set_model(opt):

    model = DualSupervisedIMUEncoder(opt)

    criterion = torch.nn.CrossEntropyLoss()

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


    for idx, ((input_data1, input_data2), labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_data1 = input_data1.cuda()
            input_data2 = input_data2.cuda()
            labels = labels.cuda()
        bsz = labels.shape[0]


        output = model(input_data1, input_data2)

        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc, _ = accuracy(output, labels, topk=(1, 5))
        top1.update(acc[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    confusion = np.zeros((opt.num_class, opt.num_class))
    label_list = []
    pred_list = []

    with torch.no_grad():
        end = time.time()
        for idx, ((input_data1, input_data2), labels) in enumerate(val_loader):

            if torch.cuda.is_available():
                input_data1 = input_data1.cuda()
                input_data2 = input_data2.cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(input_data1, input_data2)    

            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc, _ = accuracy(output, labels, topk=(1, 5))
            top1.update(acc[0], bsz)

            # calculate and store confusion matrix
            label_list.extend(labels.cpu().numpy())
            pred_list.extend(output.max(1)[1].cpu().numpy())

            rows = labels.cpu().numpy()
            cols = output.max(1)[1].cpu().numpy()

            for label_index in range(labels.shape[0]):
                confusion[rows[label_index], cols[label_index]] += 1

            # update metric
            losses.update(loss.item(), bsz)
            acc, _ = accuracy(output, labels, topk=(1, 5))
            top1.update(acc[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    F1score_test = f1_score(label_list, pred_list, average="macro") # macro sees all class with the same importance

    pprint(' * Acc@1 {top1.avg:.3f}\t'
        'F1-score {F1score_test:.3f}\t'.format(top1=top1, F1score_test=F1score_test))

    return losses.avg, top1.avg, confusion, F1score_test, label_list, pred_list


def main():

    opt = parse_evaluation_option("lowerbound_label", "lowerbound_label")
    common_modality = opt.common_modality
    other_modalities = [m for m in ['acc', 'gyro', 'mag'] if m != common_modality]
    opt.mod1 = other_modalities[0]
    opt.mod2 = other_modalities[1]

    best_acc = 0
    best_f1 = 0

    # build data loader
    train_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = optim.Adam([ 
                {'params': model.mod1_encoder.parameters(), 'lr': 1e-4},   # 0
                {'params': model.mod2_encoder.parameters(), 'lr': 1e-4},   # 0
                {'params': model.classifier.parameters(), 'lr': opt.learning_rate}],
                # momentum=opt.momentum,
                weight_decay=opt.weight_decay)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    record_acc = np.zeros(opt.epochs)
    record_f1 = np.zeros(opt.epochs)
    record_loss = np.zeros(opt.epochs)
    record_acc_train = np.zeros(opt.epochs)

    pprint(f"Start Training")
    for epoch in tqdm(range(1, opt.epochs + 1), desc=f'Epoch: ', unit='items', ncols=80, colour='green', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining}]'):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)

        val_loss, val_acc, confusion, val_F1score, label_list, pred_list = validate(test_loader, model, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_F1score

        record_loss[epoch-1] = loss
        record_acc[epoch-1] = val_acc
        record_f1[epoch-1] = val_F1score
        record_acc_train[epoch-1] = train_acc

        label_list = np.array(label_list)
        pred_list = np.array(pred_list)
        np.savetxt(os.path.join(opt.result_folder , "confusion.txt"), confusion)
        np.savetxt(os.path.join(opt.result_folder , "label.txt"), label_list)
        np.savetxt(os.path.join(opt.result_folder , "pred.txt"), pred_list)
        np.savetxt(os.path.join(opt.result_folder , "loss.txt"), record_loss)
        np.savetxt(os.path.join(opt.result_folder , "test_accuracy.txt"), record_acc)
        np.savetxt(os.path.join(opt.result_folder , "test_f1.txt"), record_f1)
        np.savetxt(os.path.join(opt.result_folder , "train_accuracy.txt"), record_acc_train)

    # save the last model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    # print("result of {}:".format(opt.dataset))
    print('best accuracy: {:.3f}'.format(best_acc))
    print('best f1: {:.3f}'.format(best_f1))
    print('last accuracy: {:.3f}'.format(val_acc))
    print('final F1:{:.3f}'.format(val_F1score))

    pprint('best accuracy: {:.3f}'.format(best_acc))
    pprint('best f1: {:.3f}'.format(best_f1))
    pprint('last accuracy: {:.3f}'.format(val_acc))
    pprint('final F1:{:.3f}'.format(val_F1score))


if __name__ == '__main__':
    main()
