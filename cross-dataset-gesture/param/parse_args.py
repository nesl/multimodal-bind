import math
import os
import random


import argparse
import numpy as np
import torch

import utils.log as log

def set_auto(opt):
    if opt.exp_type in {"lowerbound"} or opt.exp_tag in {"eval", "label_eval"}:
        opt.stage = "eval"
    else:
        opt.stage = "train"
    
    if opt.dataset == "GR4DHCI":
        opt.num_class = 8
    elif opt.dataset == "DHG":
        opt.num_class = 14
    elif opt.dataset == "Briareo":
        opt.num_class = 12
    return opt

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--exp_type', type=str, required=True)
    parser.add_argument('--exp_tag', type=str, required=True)
    # system config
    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')

    # optimization
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
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
    parser.add_argument('--model', type=str, default='MMBind')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['Briareo', 'DHG', 'GR4DHCI'], help='dataset to use')
    parser.add_argument('--num_class', type=int, default=8,
                        help='num_class')
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    # parser.add_argument('--load_pretrain', type=str, default='no_load', help='load_pretrain')

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

    # gpu setting (tommy)
    parser.add_argument('--gpu', type=int, default=-1)

    # modality setting
    parser.add_argument('--modality', type=str, default="skeleton")

    parser.add_argument('--dataset_split', type=str, default="random_split")

    parser.add_argument('--use_pair', type=bool, default=False)

    parser.add_argument("--pairing", type=bool, default=False)

    parser.add_argument("--load_pretrain", type=str, default="no_load")
    
    parser.add_argument("--label_ratio", type=float, default=1.0)

    opt = parser.parse_args()
    
    exp_type = opt.exp_type
    exp_tag = opt.exp_tag

    # Dataset
    opt.indice_file = f"./indices/{opt.dataset}/{opt.dataset_split}"

    if not os.path.exists(opt.indice_file):
        raise ValueError(f"{opt.indice_file} not found, please generate with preprocess.py/generate_index.py")
    
    opt.processed_data_path = "/home/tkimura4/multimodal-bind/cross-dataset-gesture/gesture_recog_dataset_processed"
    if not os.path.exists(opt.processed_data_path):
        raise ValueError(f"{opt.processed_data_path} not found")

    print(f"TODO: Set pretrain config here")
    # Set the seed for numpy
    # Set the seed for PyTorch
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)  # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(opt.seed)

    # opt.valid_mods = ['acc', 'gyro'] if opt.dataset == 'train_A' else ['acc', 'mag']
    
    
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    opt.save_path = f'./weights/{exp_type}/{opt.dataset}_{exp_tag}_{opt.load_pretrain}_{opt.modality}_{opt.seed}_{opt.dataset_split}_{opt.label_ratio}/'
    
    if opt.exp_tag in {"unimod", "contrastive", "label_pair", "label_contrastive"}:
        opt.save_path = f'./weights/{exp_type}/{exp_tag}_{opt.load_pretrain}_{opt.modality}_{opt.seed}_{opt.dataset_split}_{opt.label_ratio}/'
    
    print(f"TODO: Add different save path for mmbind pairs")
    # # set the path according to the environment
    # if "label" in exp_type and opt.pairing:
    #     opt.save_path = f'./{exp_type}/save_{opt.dataset}_{exp_tag}_{opt.load_pretrain}_{opt.common_modality}_{opt.seed}_{opt.dataset_split}_usepair_{opt.use_pair}_data_pairing/'
    # elif "mmbind_label" in exp_type:
    #     opt.save_path = f'./{exp_type}/save_{opt.dataset}_{exp_tag}_{opt.load_pretrain}_{opt.common_modality}_{opt.seed}_{opt.dataset_split}_usepair_{opt.use_pair}/'
    # elif "save_mmbind" in exp_type and ("unimod_autoencoder" in exp_tag or "contrastive" in exp_tag):
    #     opt.save_path = f'./{exp_type}/save_{opt.dataset}_{exp_tag}_{opt.load_pretrain}_{opt.common_modality}_{opt.seed}_{opt.dataset_split}/'
    # else:
    #     opt.save_path = f'./{exp_type}/save_{opt.dataset}_{exp_tag}_{opt.load_pretrain}_{opt.common_modality}_{opt.seed}_{opt.dataset_split}/'
    
    opt.model_path = opt.save_path + 'models'
    opt.tb_path = opt.save_path + 'tensorboard'
    opt.result_path = opt.save_path + f'results/lr_{opt.learning_rate}_decay_{opt.lr_decay_rate}_bsz_{opt.batch_size}/'

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
    opt.log_folder = os.path.join(opt.save_path, "logs")

    for folder in [opt.tb_folder, opt.save_folder, opt.result_path, opt.log_folder]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    # setup logger
    opt.log_file = os.path.join(opt.log_folder, f"train_{opt.learning_rate}_{opt.epochs}_{opt.seed}.log")
    log.init_logger(opt.log_file)
    log.divide(f"Begin Training of {exp_type} - {exp_tag}")
    log.logprint(f"Experiment random seed: {opt.seed}")
    log.logprint(f"Experiment save path: {opt.save_path}")
    log.logprint(f"Initialized log file to {opt.log_file}")
    
    opt = set_auto(opt)

    # minor gpu setting, ignore 
    if opt.gpu != -1:
        opt.device = select_gpu(opt.gpu)

    return opt 

def select_gpu(device):
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
    return device   