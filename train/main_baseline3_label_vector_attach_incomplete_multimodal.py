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
from models.imu_models import DualMaskedIMUEncoder


def load_single_modal_set_masked(opt, mod, root):

    print(f"=\tLoading masked data {mod} from {root}")
    train_dataset = data.Multimodal_masked_dataset([], [mod], root=root, opt=opt)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    
    mod1_data = []
    mod2_data = []
    y = []
    masks = []

    for _, batch in enumerate(train_loader):
        mod1_x = batch[opt.mod1]
        mod2_x = batch[opt.mod2]
        mask = batch['mask']
        labels = batch['action']

        mod1_data.append(mod1_x)
        mod2_data.append(mod2_x)
        y.append(labels)

        mask = mask[:, 1:] # index 0 is the acc, which is not considered at all
        masks.append(mask)
    
    # [nb] -> [nb * b]
    mod1_data = torch.concatenate(mod1_data, dim=0)
    mod2_data = torch.concatenate(mod2_data, dim=0)
    y = torch.concatenate(y, dim=0)
    masks = torch.concatenate(masks, dim=0)
    print(f"=\tSingle modal set ({root}) size: {mod1_data.shape}")

    return mod1_data, mod2_data, y, masks

def load_test_dataset_masked(opt, root):

    print(f"=\tLoading data {opt.mod1} and {opt.mod2} from {root}")
    # Load all, basically no mask then
    test_dataset = data.Multimodal_masked_dataset([], [opt.mod1, opt.mod2], root=root, opt=opt)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    
    x1 = []
    x2 = []
    y = []
    masks = []
    for _, batch in enumerate(test_loader):
        mod1_data = batch[opt.mod1]
        mod2_data = batch[opt.mod2]
        labels = batch['action']

        x1.append(mod1_data)
        x2.append(mod2_data)
        y.append(labels)

        mask = batch['mask'][:, 1:] # see above for why 1: is added
        masks.append(mask)
    
    # [nb] -> [nb * b]
    x1 = torch.concatenate(x1, dim=0)
    x2 = torch.concatenate(x2, dim=0)
    y = torch.concatenate(y, dim=0)
    masks = torch.concatenate(masks, dim=0)

    return x1, x2, y, masks

def load_dataset_masked(opt):
    x1_A, x2_A, y_A, mask_A = load_single_modal_set_masked(opt, opt.mod1, "train_A")
    x1_B, x2_B, y_B, mask_B = load_single_modal_set_masked(opt, opt.mod2, "train_B")
    
    x1 = np.vstack((x1_A, x1_B))
    x2 = np.vstack((x2_A, x2_B))

    y = np.hstack((y_A, y_B))
    mask = np.vstack((mask_A, mask_B))
    
    return x1, x2, y, mask

def set_loader(opt):
    if opt.pairing:
        x_train_1, x_train_2, y_train, mask_train = load_test_dataset_masked(opt, "train_C")
    else:
        x_train_1, x_train_2, y_train, mask_train = load_dataset_masked(opt)

    x_test_1, x_test_2, y_test, mask_test = load_test_dataset_masked(opt, "test")

    train_dataset = data.Multimodal_incomplete_dataset_direct_load(x_train_1, x_train_2, y_train, mask_train)
    test_dataset = data.Multimodal_incomplete_dataset_direct_load(x_test_1, x_test_2,  y_test, mask_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)

    return train_loader, val_loader

def set_model(opt):

    model = DualMaskedIMUEncoder(opt)
    criterion = torch.nn.CrossEntropyLoss()

    if opt.pairing:
        weight = f"./save_baseline3_label/save_train_AB_vector_attach_incomplete_multimodal_no_load_{opt.common_modality}_{opt.seed}_{opt.dataset_split}/models/lr_0.0001_decay_0.0001_bsz_64/last.pth"
        print(f"=\tLoading model pretrain weights from {weight}")
        model.load_state_dict(torch.load(weight)['model'])

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

    label_list = []
    pred1_list = []

    for idx, (input_data1, input_data2, labels, mask) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_data1 = input_data1.cuda()
            input_data2 = input_data2.cuda()
            labels = labels.cuda()
            mask = mask.cuda()
        bsz = input_data1.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        output = model(input_data1, input_data2, mask)


        label_list.extend(labels.cpu().numpy())
        pred1_list.extend(output.max(1)[1].cpu().numpy())
        
        loss = criterion(output, labels)

        acc, _ = accuracy(output, labels, topk=(1, 5))

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

    F1score = f1_score(label_list, pred1_list, average=None)

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    confusion = np.zeros((opt.num_class, opt.num_class))

    label_list = []
    pred_list = []

    with torch.no_grad():
        end = time.time()
        for idx, (input_data1, input_data2, labels, mask) in enumerate(val_loader):
            if torch.cuda.is_available():
                input_data1 = input_data1.cuda()
                input_data2 = input_data2.cuda()
                labels = labels.cuda()
                mask = mask.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(input_data1, input_data2, mask)

            label_list.extend(labels.cpu().numpy())
            pred_list.extend(output.max(1)[1].cpu().numpy())

            loss = criterion(output, labels)

            rows = labels.cpu().numpy()
            cols = output.max(1)[1].cpu().numpy()

            for label_index in range(labels.shape[0]):
                confusion[rows[label_index], cols[label_index]] += 1

            acc, _ = accuracy(output, labels, topk=(1, 5))

            losses.update(loss.item(), bsz)
            top1.update(acc[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    F1score_test = f1_score(label_list, pred_list, average="macro") # macro sees all class with the same importance

    pprint(' * Acc@1 {top1.avg:.3f}\t'
        'F1-score {F1score_test:.3f}\t'.format(top1=top1, F1score_test=F1score_test))

    return losses.avg, top1.avg, confusion, F1score_test, label_list, pred_list


def main():
    opt = parse_option("save_baseline3_label", "vector_attach_incomplete_multimodal")
    common_modality = opt.common_modality
    other_modalities = [m for m in ['acc', 'gyro', 'mag'] if m != common_modality]
    opt.mod1 = other_modalities[0]
    opt.mod2 = other_modalities[1]

    pprint(f"Common modality: {opt.common_modality}")
    print(f"=\tCommon modality: {opt.common_modality}")

    best_acc = 0
    best_f1 = 0

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    record_loss = np.zeros(opt.epochs)
    record_acc = np.zeros(opt.epochs)
    record_f1 = np.zeros(opt.epochs)
    record_acc_train = np.zeros(opt.epochs)

    pprint(f"Start Training")
    for epoch in tqdm(range(1, opt.epochs + 1), desc=f'Epoch: ', unit='items', ncols=80, colour='green', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining}]'):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)

        # evaluation
        val_loss, val_acc, confusion, val_F1score, label_list, pred_list = validate(val_loader, model, criterion, opt)


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
