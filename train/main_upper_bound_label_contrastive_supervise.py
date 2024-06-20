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

from tqdm import tqdm
from modules.option_utils import parse_option
from modules.print_utils import pprint

from models.imu_models import DualContrastiveIMUEncoder
from shared_files.contrastive_design import ConFusionLoss


def load_dual_dataset(opt, root):
    """Load dual mod (mod1, mod2 -- excluding the common modality) dataset 
    """
    print(f"=\tLoading data {opt.mod1} and {opt.mod2} from {root}")
    dataset = data.Multimodal_dataset([], [opt.mod1, opt.mod2], root=root, opt=opt)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    
    x1 = []
    x2 = []
    y = []
    for _, batch in enumerate(dataloader):
        mod1_data = batch[opt.mod1]
        mod2_data = batch[opt.mod2]
        labels = batch['action']

        x1.append(mod1_data)
        x2.append(mod2_data)
        y.append(labels)
    
    # [nb] -> [nb * b]
    x1 = torch.concatenate(x1, dim=0)
    x2 = torch.concatenate(x2, dim=0)
    y = torch.concatenate(y, dim=0)

    print(f"=\t")
    return x1, x2, y

def set_loader(opt):

    #load labeled train and test data

    x1_train_A, x2_train_A, y_train_A = load_dual_dataset(opt, "train_A") # load mod 1 and mod 2 from train_A
    x1_train_B, x2_train_B, y_train_B = load_dual_dataset(opt, "train_B") # load mod 1 and mod 2 from train_B

    x1_train  = torch.concat((x1_train_A, x1_train_B), dim=0) # [b1, dim] + [b2, dim] -> [b, dim]
    x2_train = torch.concat((x2_train_A, x2_train_B), dim=0) # [b1, dim] + [b2, dim] -> [b, dim]
    y_train = torch.concat((y_train_A, y_train_B), dim=0)

    x_test_1, x_test_2, y_test = load_dual_dataset(opt, "test")

    train_dataset = data.Multimodal_dataset_direct_load(x1_train, x2_train, y_train)
    test_dataset = data.Multimodal_dataset_direct_load(x_test_1, x_test_2, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)

    return train_loader, val_loader


def set_model(opt):

    # model = DualContrastiveIMUEncoder(input_size=1, num_classes=opt.num_class)
    model = DualContrastiveIMUEncoder(opt)

    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = ConFusionLoss(temperature=opt.temp)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()
        cudnn.benchmark = True
        
    return model, criterion1, criterion2



def train(train_loader, model, criterion1, criterion2, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()


    for idx, (input_data1, input_data2, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_data1 = input_data1.cuda()
            input_data2 = input_data2.cuda()
            labels = labels.cuda()
        bsz = labels.shape[0]


        output, feature1, feature2 = model(input_data1, input_data2)

        features = torch.stack([feature1, feature2], dim=1)

        loss1 = criterion1(output, labels)
        loss2 = criterion2(features)
        loss = loss1 + loss2

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
        for idx, (input_data1, input_data2, labels) in enumerate(val_loader):

            if torch.cuda.is_available():
                input_data1 = input_data1.cuda()
                input_data2 = input_data2.cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output, feature1, feature2 = model(input_data1, input_data2)

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

    opt = parse_option("save_upperbound_label", "contrastive_supervise")
    common_modality = opt.common_modality
    other_modalities = [m for m in ['acc', 'gyro', 'mag'] if m != common_modality]
    opt.mod1 = other_modalities[0]
    opt.mod2 = other_modalities[1]

    pprint(f"Common modality: {opt.common_modality}")
    print(f"=\tCommon modality: {opt.common_modality}")

    best_acc = 0
    best_f1 = 0

    # build data loader
    train_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, criterion1, criterion2 = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    record_acc = np.zeros(opt.epochs)
    record_f1 = np.zeros(opt.epochs)
    record_loss = np.zeros(opt.epochs)
    record_acc_train = np.zeros(opt.epochs)

    pprint(f"Start Training")
    for epoch in tqdm(range(1, opt.epochs + 1), desc=f'Epoch: ', unit='items', ncols=80, colour='green', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining}]'):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        loss, train_acc = train(train_loader, model, criterion1, criterion2, optimizer, epoch, opt)

        # eval for one epoch
        val_loss, val_acc, confusion, val_F1score, label_list, pred_list = validate(test_loader, model, criterion1, opt)
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_F1score

        record_loss[epoch-1] = loss
        record_acc[epoch-1] = val_acc
        record_f1[epoch-1] = val_F1score
        record_acc_train[epoch-1] = train_acc

        label_list = np.array(label_list)
        pred_list = np.array(pred_list)
        np.savetxt(opt.result_path+ "confusion.txt", confusion)
        np.savetxt(opt.result_path + "label.txt", label_list)
        np.savetxt(opt.result_path + "pred.txt", pred_list)
        np.savetxt(opt.result_path + "loss.txt", record_loss)
        np.savetxt(opt.result_path + "test_accuracy.txt", record_acc)
        np.savetxt(opt.result_path + "test_f1.txt", record_f1)
        np.savetxt(opt.result_path + "train_accuracy.txt", record_acc_train)

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
