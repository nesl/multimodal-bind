import math
import os
import sys

import argparse
import numpy as np
import torch


from modules.print_utils import init_logger, pprint


def parse_evaluation_option(exp_type, exp_tag):
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
    parser.add_argument('--weight_decay', type=float, default=5e-4,
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
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--common_modality', type=str, default="acc")
    parser.add_argument('--dataset_split', type=str, default="split_0")

    parser.add_argument('--load_pretrain', type=str, default="no_load")

    opt = parser.parse_args()

    print()
    print(f"="*80)
    print(f"=\tBegin Training of {exp_type} - {exp_tag}")




    opt.indice_file = f"../indices/{opt.dataset_split}"
    if not os.path.exists(opt.indice_file):
        raise ValueError(f"{opt.indice_file} not found, please generate with preprocess.py/generate_index.py")
    opt.processed_data_path = "../processed_data_all" if "label" in opt.dataset_split else "../processed_data"
    if not os.path.exists(opt.processed_data_path):
        raise ValueError(f"{opt.processed_data_path} not found")


    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # set the path according to the environment
    # opt.save_path = f'./save_{opt.dataset}{exp_tag}/'
    opt.save_path = f'./save_{opt.dataset}{exp_tag}_{opt.common_modality}_{opt.seed}_{opt.dataset_split}_{opt.load_pretrain}/'
    
    print(f"=\tSaving models to path {opt.save_path}")
    print(f"=\tLearning rate: {opt.learning_rate}")
    print(f"=\tWeight decay: {opt.weight_decay}")
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
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    opt.result_folder = os.path.join(opt.result_path, opt.model_name)
    opt.log_folder = os.path.join(opt.save_path, "logs")

    for folder in [opt.tb_folder, opt.save_folder, opt.result_path, opt.log_folder, opt.result_folder]:
        if not os.path.isdir(folder):
            os.makedirs(folder)
        
    # setup logger
    opt.log_file = os.path.join(opt.log_folder, f"train_lr_{opt.learning_rate}_decay_{opt.weight_decay}_{opt.epochs}.log")
    init_logger(opt.log_file)
    pprint(f"Initialized log file to {opt.log_file}")

    # minor gpu setting, ignore 
    if opt.gpu != -1:
        select_gpu(opt.gpu)
    
    opt.exp_tag = exp_tag

    return opt

def select_gpu(device):
    s = f"Torch-{torch.__version__} "
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"