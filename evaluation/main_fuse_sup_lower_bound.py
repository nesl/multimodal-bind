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

from models.imu_models import SupervisedGyroMag


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=5e-4,
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
    parser.add_argument('--dataset', type=str, default='train_C/label_216/', help='dataset')
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
    opt.save_path = './save_{}supfuse_lower_bound/'.format(opt.dataset)
    opt.model_path = opt.save_path + 'models'
    opt.tb_path = opt.save_path + 'tensorboard'.format(opt.dataset)
    opt.result_path = opt.save_path + 'results/'.format(opt.dataset)

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

    train_dataset = data.Multimodal_dataset([], ['acc', 'gyro', 'mag'], root='../../PAMAP_Dataset/trainC/')
    test_dataset = data.Multimodal_dataset([], ['acc', 'gyro', 'mag'], root='../../PAMAP_Dataset/test/')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)

    return train_loader, val_loader


def set_model(opt):

    model = SupervisedGyroMag()
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

    for idx, batched_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        labels = batched_data['action']

        if torch.cuda.is_available():
            labels = labels.cuda()
        bsz = labels.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        output = model(batched_data)
        #print(output)
        

        label_list.extend(labels.cpu().numpy())
        pred1_list.extend(output.max(1)[1].cpu().numpy())
        
        loss = criterion(output, labels)

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
        for idx, batched_data in enumerate(val_loader):

            labels = batched_data['action']
            if torch.cuda.is_available():
                labels = labels.cuda()
            bsz = labels.shape[0]

            # warm-up learning rate
            # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

            output = model(batched_data)

            label_list.extend(labels.cpu().numpy())
            pred_list.extend(output.max(1)[1].cpu().numpy())


            loss = criterion(output, labels)

            rows = labels.cpu().numpy()
            cols = output.max(1)[1].cpu().numpy()

            # print("labels:",rows)
            # print("output:",cols)
            # print("old confusion:",confusion)

            for label_index in range(labels.shape[0]):
                confusion[rows[label_index], cols[label_index]] += 1

            # update metric
            # acc, rec, prec, target_positive, pred_positive = rate_eval(output, labels, opt.num_class, topk=(1, 5))
            acc, _ = accuracy(output, labels, topk=(1, 5))
            # print("Compare acc, rec:", acc, rec)

            losses.update(loss.item(), bsz)
            top1.update(acc[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))


    # print("test-f1-score", f1_score(label_list, pred_list, average=None))

    F1score_test = f1_score(label_list, pred_list, average="macro") # macro sees all class with the same importance

    print(' * Acc@1 {top1.avg:.3f}\t'
        'F1-score {F1score_test:.3f}\t'.format(top1=top1, F1score_test=F1score_test))

    return losses.avg, top1.avg, confusion, F1score_test, label_list, pred_list


def main():
    best_acc = 0
    best_rec = 0
    best_prec = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    record_loss = np.zeros(opt.epochs)
    record_acc = np.zeros(opt.epochs)
    record_f1 = np.zeros(opt.epochs)
    record_acc_train = np.zeros(opt.epochs)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()

        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        # logger.log_value('train_loss', loss, epoch)
        # logger.log_value('train_acc', train_acc, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        loss, val_acc, confusion, val_F1score, label_list, pred_list = validate(val_loader, model, criterion, opt)
        # logger.log_value('val_loss', loss, epoch)
        # logger.log_value('val_acc', val_acc, epoch)
        # logger.log_value('val_f1', val_F1score, epoch)

        # val_F1 = 2 * val_rec * val_prec / (val_rec + val_prec)

        if val_acc > best_acc:
            best_acc = val_acc

        record_loss[epoch-1] = loss
        record_acc[epoch-1] = val_acc
        record_f1[epoch-1] = val_F1score
        record_acc_train[epoch-1] = train_acc
        # if epoch % opt.save_freq == 0:
        #     save_file = os.path.join(
        #         opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     save_model(model, optimizer, opt, epoch, save_file)

        label_list = np.array(label_list)
        pred_list = np.array(pred_list)
        np.savetxt(opt.result_path+ "confusion.txt", confusion)
        np.savetxt(opt.result_path + "label.txt", label_list)
        np.savetxt(opt.result_path + "pred.txt", pred_list)
        np.savetxt(opt.result_path + "loss.txt", record_loss)
        np.savetxt(opt.result_path + "test_accuracy.txt", record_acc)
        np.savetxt(opt.result_path + "test_f1.txt", record_f1)
        np.savetxt(opt.result_path + "train_accuracy.txt", record_acc_train)
    
    # save the last model
    # save_file = os.path.join(
    #     opt.save_folder, 'last.pth')
    # save_model(model, optimizer, opt, opt.epochs, save_file)

    # print("result of {}:".format(opt.dataset))
    print('best accuracy: {:.3f}'.format(best_acc))
    print('last accuracy: {:.3f}'.format(val_acc))
    print('final F1:{:.3f}'.format(val_F1score))
    # print("confusion:{:.3f}", confusion)



if __name__ == '__main__':
    main()
