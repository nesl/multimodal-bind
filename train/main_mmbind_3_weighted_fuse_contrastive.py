import os
import time

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn

import numpy as np

from shared_files.util import AverageMeter
from shared_files.util import adjust_learning_rate
from shared_files.util import set_optimizer, save_model
from shared_files import data_pre as data

from models.imu_models import GyroMagEncoder, SingleIMUAutoencoder, ModEncoder
from shared_files.contrastive_weighted_design import FeatureConstructor, ConFusionLoss

from tqdm import tqdm
from modules.option_utils import parse_option
from modules.print_utils import pprint


def set_loader(opt):
    print(f"=\tInitializing Dataloader")

    dataset = f"./save_mmbind/train_all_paired_AB_{opt.common_modality}_{opt.seed}_{opt.dataset_split}/"
    pprint(f"=\tLoading dataset from {dataset}")
    print(f"=\tLoading dataset from {dataset}")
    train_dataset = data.Multimodal_dataset([], ['acc', 'gyro', 'mag'], root=dataset, opt=opt)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True, drop_last=True)

    return train_loader



def set_model(opt):
    print(f"=\tInitializing Backbone models")
    model = ModEncoder()
    criterion = ConFusionLoss(temperature=opt.temp)

    common_modality = opt.common_modality
    other_modalities = [m for m in ['acc', 'gyro', 'mag'] if m != common_modality]

 
    if opt.load_pretrain == "load_pretrain":
        model_template = SingleIMUAutoencoder('acc')
        model_template.load_state_dict(torch.load(f'./save_baseline1/save_train_A_autoencoder_no_load_gyro_{opt.seed}_{opt.dataset_split}/models/lr_0.0001_decay_0.0001_bsz_64/last.pth')['model'])
        model.gyro_encoder = model_template.encoder
        model_template.load_state_dict(torch.load(f'./save_baseline1/save_train_B_autoencoder_no_load_mag_{opt.seed}_{opt.dataset_split}/models/lr_0.0001_decay_0.0001_bsz_64/last.pth')['model'])
        model.mag_encoder = model_template.encoder

    # # enable synchronized Batch Normalization
    # if opt.syncBN:
    #     model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
        #     model = torch.nn.DataParallel(model)
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

    end = time.time()

    common_modality = opt.common_modality
    other_modalities = [m for m in ['acc', 'gyro', 'mag'] if m != common_modality]

    for idx, batched_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        similarity = batched_data["similarity"].cuda()

      

        acc_embed, gyro_embed, mag_embed = model(batched_data)

        bsz = gyro_embed.shape[0]

        features = torch.stack([acc_embed, gyro_embed, mag_embed], dim=1)

        loss = criterion(features, similarity)
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


def main():
    opt = parse_option("save_mmbind", "weighted_contrastive")

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
    pprint(f"Start Training")
    for epoch in tqdm(range(1, opt.epochs + 1), desc=f'Epoch: ', unit='items', ncols=80, colour='green', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining}]'):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)

        record_loss[epoch-1] = loss

        pprint(f"Epoch: {epoch} - Loss: {loss}")
    
    np.savetxt(opt.result_path + f"loss_{opt.learning_rate}_{opt.epochs}.txt", record_loss)
    
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)



if __name__ == '__main__':
    main()
