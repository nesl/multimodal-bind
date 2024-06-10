import os
import time

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

from models.imu_models import SingleIMUAutoencoder, SupervisedAccGyro, SupervisedAccMag, SupervisedGyroMag
from modules.option_utils import parse_evaluation_option
from modules.print_utils import pprint

from tqdm import tqdm

def set_loader(opt):

    train_dataset = data.Multimodal_dataset([], ['acc', 'gyro', 'mag'], root='train_C', opt=opt)
    test_dataset = data.Multimodal_dataset([], ['acc', 'gyro', 'mag'], root='test', opt=opt)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)

    return train_loader, val_loader



def set_model(opt):

    mod = opt.common_modality
    mod_space = ['acc', 'gyro', 'mag']
    multi_mod_space = [[mod, m] for m in mod_space if m != mod]

    opt.valid_mod = multi_mod_space

    mod1, mod2 = opt.valid_mod[0][1], opt.valid_mod[1][1]

    if 'gyro' in {mod1, mod2} and 'mag' in {mod1, mod2}:
        print(f"=\tInitializing GyroMag model")
        model = SupervisedGyroMag()
    
    if 'acc' in {mod1, mod2} and 'mag' in {mod1, mod2}:
        print(f"=\tInitializing AccMag model")
        model = SupervisedAccMag()
    
    if 'acc' in {mod1, mod2} and 'gyro' in {mod1, mod2}:
        print(f"=\tInitializing AccGyro model")
        model = SupervisedAccGyro()
    
    model_template = SingleIMUAutoencoder('mag') # any mod should be ok

    mod1_weight = f"../train/save_baseline1/save_train_A_autoencoder_no_load_{mod1}_{opt.seed}_{opt.dataset_split}/models/lr_0.0001_decay_0.0001_bsz_64/last.pth"
    mod2_weight = f"../train/save_baseline1/save_train_B_autoencoder_no_load_{mod2}_{opt.seed}_{opt.dataset_split}/models/lr_0.0001_decay_0.0001_bsz_64/last.pth"

    pprint(f"=\tLoad {mod1} weight from {mod1_weight}")
    pprint(f"=\tLoad {mod2} weight from {mod2_weight}")
    model_template.load_state_dict(torch.load(mod1_weight)['model'])
    setattr(model, f"{mod1}_encoder", model_template.encoder)
    # model.gyro_encoder = model_template.encoder
    model_template.load_state_dict(torch.load(mod2_weight)['model'])
    setattr(model, f"{mod2}_encoder", model_template.encoder)
    # model.mag_encoder = model_template.encoder

    criterion = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization

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

    for idx, batched_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        labels = batched_data['action']

        if torch.cuda.is_available():
            labels = labels.cuda()
        bsz = labels.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        output = model(batched_data)


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
        # if (idx + 1) % opt.print_freq == 0:
        #     print('Train: [{0}][{1}/{2}]\t'
        #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'loss {loss.val:.3f} ({loss.avg:.3f})\t'
        #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
        #            epoch, idx + 1, len(train_loader), batch_time=batch_time,
        #            data_time=data_time, loss=losses, top1=top1))
        #     sys.stdout.flush()

    # print("label_list:",label_list, len(label_list))
    # print("pred1_list:",pred1_list)

    F1score = f1_score(label_list, pred1_list, average=None)
    pprint(f'feature_f1: {F1score}')


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


    # print("test-f1-score", f1_score(label_list, pred_list, average=None))

    F1score_test = f1_score(label_list, pred_list, average="macro") # macro sees all class with the same importance

    pprint(' * Acc@1 {top1.avg:.3f}\t'
        'F1-score {F1score_test:.3f}\t'.format(top1=top1, F1score_test=F1score_test))

    return losses.avg, top1.avg, confusion, F1score_test, label_list, pred_list


def main():
    best_acc = 0
    best_rec = 0
    best_prec = 0
    opt = parse_evaluation_option(exp_type="baseline1", exp_tag="baseline1")

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    # optimizer = set_optimizer(opt, model)

    # build optimizer
    optimizer = optim.Adam(model.parameters(),
                # momentum=opt.momentum,
                weight_decay=opt.weight_decay)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    record_loss = np.zeros(opt.epochs)
    record_acc = np.zeros(opt.epochs)
    record_f1 = np.zeros(opt.epochs)
    record_acc_train = np.zeros(opt.epochs)

    # training routine
    pprint(f"Start Training")
    for epoch in tqdm(range(1, opt.epochs + 1), desc=f'Epoch: ', unit='items', ncols=80, colour='green', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining}]'):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)

        # evaluation
        val_loss, val_acc, confusion, val_F1score, label_list, pred_list = validate(val_loader, model, criterion, opt)


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
    
        np.savetxt(os.path.join(opt.result_folder , "confusion.txt"), confusion)
        np.savetxt(os.path.join(opt.result_folder , "label.txt"), label_list)
        np.savetxt(os.path.join(opt.result_folder , "pred.txt"), pred_list)
        np.savetxt(os.path.join(opt.result_folder , "loss.txt"), record_loss)
        np.savetxt(os.path.join(opt.result_folder , "test_accuracy.txt"), record_acc)
        np.savetxt(os.path.join(opt.result_folder , "test_f1.txt"), record_f1)
        np.savetxt(os.path.join(opt.result_folder , "train_accuracy.txt"), record_acc_train)
    
    # save the last model
    # save_file = os.path.join(
    #     opt.save_folder, 'last.pth')
    # save_model(model, optimizer, opt, opt.epochs, save_file)

    # print("result of {}:".format(opt.dataset))
    print('best accuracy: {:.3f}'.format(best_acc))
    print('last accuracy: {:.3f}'.format(val_acc))
    print('final F1:{:.3f}'.format(val_F1score))

    pprint('best accuracy: {:.3f}'.format(best_acc))
    pprint('last accuracy: {:.3f}'.format(val_acc))
    pprint('final F1:{:.3f}'.format(val_F1score))



if __name__ == '__main__':
    main()
