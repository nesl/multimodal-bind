import os
import torch
import numpy as np
import time
from tqdm import tqdm

import utils.log as log
from utils.evaluation import AverageMeter, accuracy, adjust_learning_rate, save_model
from sklearn.metrics import f1_score

def calc_loss(opt, model, data, labels, loss_func):
    if opt.exp_type == "mmbind":
        if opt.exp_tag == "unimod":
            mod = opt.modality
            pred = model(data)
            gt = data[mod]
            return loss_func(pred, gt)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
            
            
def val_loop(opt, epoch, model, loss_func, dataloader, optimizer):
    model.eval()
    losses = AverageMeter()
    with torch.no_grad():    
        for idx, (data, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):            
            for mod in data:
                data[mod] = data[mod].cuda()
            labels = labels.cuda()
            loss = calc_loss(opt, model, data, labels, loss_func)
            losses.update(loss.item(), labels.shape[0])
    return losses.avg

def train_loop(opt, epoch, model, loss_func, dataloader, optimizer):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    for idx, (data, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        data_time.update(time.time() - end)
        for mod in data:
            data[mod] = data[mod].cuda()
        labels = labels.cuda()
        
        loss = calc_loss(opt, model, data, labels, loss_func)
        losses.update(loss.item(), labels.shape[0])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
    return losses.avg
    
    
def train_engine(opt, model, loss_func, train_dataloader, val_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    record_loss = np.zeros(opt.epochs)
    log.divide(f"Start {opt.exp_type} - {opt.exp_tag} Pretraining")
    best_val_loss = np.inf
    
    for epoch in range(0, opt.epochs):
        log.divide(f"Epoch {epoch + 1}/{opt.epochs}")
        adjust_learning_rate(opt, optimizer, epoch)
        train_loss = train_loop(opt, epoch, model, loss_func, train_dataloader, optimizer)
        record_loss[epoch] = train_loss
        
        if (epoch) % opt.save_freq == 0:
            val_loss = val_loop(opt, epoch, model, loss_func, val_dataloader, optimizer)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_file = os.path.join(opt.save_folder, "best.pth")
                save_model(model, optimizer, opt, opt.epochs, save_file)
    
    save_file = os.path.join(opt.save_folder, "last.pth")
    save_model(model, optimizer, opt, opt.epochs, save_file)