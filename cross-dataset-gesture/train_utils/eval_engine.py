import os
import torch
import numpy as np
import time
from tqdm import tqdm

import utils.log as log
from utils.evaluation import AverageMeter, accuracy, adjust_learning_rate, save_model
from sklearn.metrics import f1_score


def eval_loop(opt, epoch, model, loss_func, dataloader, optimizer):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    label_list = []
    pred_list = []
    
    for idx, (data, labels, file_id) in tqdm(enumerate(dataloader), total=len(dataloader)):
        data_time.update(time.time() - end)

        # move to device
        for mod in data:
            data[mod] = data[mod].cuda()
        labels = labels.cuda()

        bsz = labels.shape[0]

        logits = model(data)

        label_list.extend(labels.cpu().numpy())
        pred_list.extend(logits.max(1)[1].cpu().numpy())

        loss = loss_func(logits, labels)

        acc, _ = accuracy(logits, labels, topk=(1, 5))

        losses.update(loss.item(), bsz)
        top1.update(acc[0], bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, top1.avg

def validate(opt, val_loader, model, criterion):
    """validation"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    confusion = np.zeros((opt.num_class, opt.num_class)).astype(int)

    label_list = []
    pred_list = []

    with torch.no_grad():
        for idx, (data, labels, file_id) in tqdm(enumerate(val_loader), total=len(val_loader)):

            # move to device
            for mod in data:
                data[mod] = data[mod].cuda()
            labels = labels.cuda()

            bsz = labels.shape[0]

            logits = model(data)

            label_list.extend(labels.cpu().numpy())
            pred_list.extend(logits.max(1)[1].cpu().numpy())

            loss = criterion(logits, labels)

            rows = labels.cpu().numpy()
            cols = logits.max(1)[1].cpu().numpy()

            for label_index in range(labels.shape[0]):
                confusion[rows[label_index], cols[label_index]] += 1

            # update metric
            acc, _ = accuracy(logits, labels, topk=(1, 5))

            losses.update(loss.item(), bsz)
            top1.update(acc[0], bsz)

    F1score_test = f1_score(label_list, pred_list, average="macro") # macro sees all class with the same importance

    return losses.avg, top1.avg, confusion, F1score_test, label_list, pred_list

def eval_engine(opt, model, loss_func, train_dataloader, val_dataloader):
    best_acc = 0
    best_f1 = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    record_loss = np.zeros(opt.epochs)
    record_acc = np.zeros(opt.epochs)
    record_f1 = np.zeros(opt.epochs)
    record_acc_train = np.zeros(opt.epochs)

    log.divide("Start Supervised Evaluation Training")
    for epoch in range(0, opt.epochs):
        log.divide(f"Epoch {epoch + 1}/{opt.epochs}")
        adjust_learning_rate(opt, optimizer, epoch)
        # train for one epoch
        loss, train_acc = eval_loop(opt, epoch, model, loss_func, train_dataloader, optimizer)
        
        log.logprint(f"Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}")
        
        if epoch % 10 == 0:
            val_loss, val_acc, confusion, val_F1score, label_list, pred_list = validate(opt, val_dataloader, model, loss_func)

            log.logprint(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, F1: {val_F1score:.4f}")
            print(confusion)
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_f1 = val_F1score

            record_loss[epoch - 1] = val_loss
            record_acc[epoch - 1] = val_acc
            record_f1[epoch - 1] = val_F1score
            record_acc_train[epoch - 1] = train_acc

            label_list = np.array(label_list)
            pred_list = np.array(pred_list)
            np.savetxt(opt.result_path + "confusion.txt", confusion)
            np.savetxt(opt.result_path + "label.txt", label_list)
            np.savetxt(opt.result_path + "pred.txt", pred_list)
            np.savetxt(opt.result_path + "loss.txt", record_loss)
            np.savetxt(opt.result_path + "test_accuracy.txt", record_acc)
            np.savetxt(opt.result_path + "test_f1.txt", record_f1)
            np.savetxt(opt.result_path + "train_accuracy.txt", record_acc_train)
    # save the last model
    save_file = os.path.join(opt.save_folder, "last.pth")
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print("best accuracy: {:.3f}".format(best_acc))
    print("best F1:{:.3f}".format(best_f1))
    print("last accuracy: {:.3f}".format(val_acc))
    print("final F1:{:.3f}".format(val_F1score))
