import os
import torch
import numpy as np
import time
from tqdm import tqdm

import utils.log as log
from utils.evaluation import AverageMeter, adjust_learning_rate, save_model
from train_utils.calc_loss import calc_loss
            
            
def val_loop(opt, epoch, model, loss_func, dataloader, optimizer, return_features=False):
    model.eval()
    losses = AverageMeter()
    all_features = []
    all_labels = []
    all_ids = []
    with torch.no_grad():    
        for idx, (data, labels, file_id) in tqdm(enumerate(dataloader), total=len(dataloader)):            
            for mod in data:
                data[mod] = data[mod].cuda()
            labels = labels.cuda()
            embedding, loss = calc_loss(opt, model, data, labels, loss_func)
            if return_features:
                all_features.append(embedding.cpu())
                all_labels.append(labels.cpu())
                all_ids.append(file_id)
            losses.update(loss.item(), labels.shape[0])
    
    if return_features:
        return all_features, all_labels, all_ids
    else:
        return losses.avg

def train_loop(opt, epoch, model, loss_func, dataloader, optimizer):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    for idx, (data, labels, file_id) in tqdm(enumerate(dataloader), total=len(dataloader)):
        data_time.update(time.time() - end)
        for mod in data:
            data[mod] = data[mod].cuda()
        labels = labels.cuda()
        
        _, loss = calc_loss(opt, model, data, labels, loss_func)
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
        log.logprint(f"Training Loss: {train_loss}")
        record_loss[epoch] = train_loss
        
        if (epoch) % 10 == 0 and val_dataloader is not None:
            val_loss = val_loop(opt, epoch, model, loss_func, val_dataloader, optimizer)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_file = os.path.join(opt.save_folder, "best.pth")
                save_model(model, optimizer, opt, opt.epochs, save_file)
            log.logprint(f"Validation Loss: {val_loss}")
        if epoch % 10 == 0:
            save_file = os.path.join(opt.save_folder, "last.pth")
            save_model(model, optimizer, opt, epoch, save_file)
        np.savetxt(os.path.join(opt.save_folder, "loss.txt"), record_loss)
    
    save_file = os.path.join(opt.save_folder, "last.pth")
    save_model(model, optimizer, opt, opt.epochs, save_file)