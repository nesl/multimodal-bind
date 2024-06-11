import os
import time

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
# from torchvision import transforms, datasets

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from shared_files.util import AverageMeter
from shared_files.util import adjust_learning_rate
from shared_files.util import set_optimizer, save_model
from shared_files import data_pre as data

from models.imu_models import GyroMagEncoder, SingleIMUAutoencoder, ModEncoder
from shared_files.contrastive_design import FeatureConstructor, ConFusionLoss

from tqdm import tqdm
from modules.option_utils import parse_option
from modules.print_utils import pprint


def set_loader(opt):
    print(f"=\tInitializing Dataloader")
    #load labeled train and test data

    dataset = f"./save_mmbind/train_all_paired_AB_{opt.common_modality}_{opt.seed}_{opt.dataset_split}/"
    pprint(f"=\tLoading dataset from {dataset}")
    print(f"=\tLoading dataset from {dataset}")
    train_dataset = data.Multimodal_dataset([], ['acc', 'gyro', 'mag'], root=dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True, drop_last=True)

    return train_loader



def set_model(opt):
    print(f"=\tInitializing Backbone models")
    # model = GyroMagEncoder()
    model = ModEncoder()
    criterion = ConFusionLoss(temperature=opt.temp)

    common_modality = opt.common_modality
    other_modalities = [m for m in ['acc', 'gyro', 'mag'] if m != common_modality]

 
    if opt.load_pretrain == "load_pretrain":
        model_template = SingleIMUAutoencoder('acc')
        model_template.load_state_dict(torch.load('./save_baseline1/save_train_A_autoencoder/models/lr_0.005_decay_0.0001_bsz_64/last.pth')['model'])
        model.gyro_encoder = model_template.encoder
        model_template.load_state_dict(torch.load('./save_baseline1/save_train_B_autoencoder/models/lr_0.005_decay_0.0001_bsz_64/last.pth')['model'])
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

      

        acc_embed, gyro_embed, mag_embed = model(batched_data)

        embed = {
            "acc": acc_embed,
            "gyro": gyro_embed,
            "mag": mag_embed
        }

        bsz = gyro_embed.shape[0]

        embed1 = embed[other_modalities[0]]
        embed2 = embed[other_modalities[1]]
        features = FeatureConstructor(embed1, embed2, 2)

        loss = criterion(features)
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
    opt = parse_option("save_mmbind", "contrastive")

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


        # if epoch % opt.save_freq == 0:
        #     save_file = os.path.join(
        #         opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     save_model(model, optimizer, opt, epoch, save_file)
    
    np.savetxt(opt.result_path + f"loss_{opt.learning_rate}_{opt.epochs}.txt", record_loss)
    
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)



if __name__ == '__main__':
    main()
