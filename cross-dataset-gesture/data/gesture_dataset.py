import os
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

import utils.log as log

def init_dataset(opt, mode="train"):
    opt.full_modalities = ["skeleton", "stereo_ir", "depth"]
    if opt.stage == "eval" or opt.exp_tag == "eval":
        opt.dataset = "Briareo"
        return MultimodalDataset(opt, mode)
    else:
        if opt.exp_type == "mmbind":
            if opt.exp_tag == "unimod":
                opt.full_modalities = [opt.modality]
                opt.dataset = "GR4DHCI"
                dataset_1 = MultimodalDataset(opt, mode)
                opt.dataset = "DHG"
                dataset_2 = MultimodalDataset(opt, mode)
                log.logprint(f"{opt.exp_type} {opt.exp_tag} - Merging GR4DHCI and DHG datasets for {opt.modality}")
                return ConcatDataset([dataset_1, dataset_2])
            elif opt.exp_tag == "pair":
                opt.original_dataset = opt.dataset
                
                opt.dataset = "GR4DHCI"
                opt.full_modalities = [opt.modality, "stereo_ir"]
                dataset_1 = MultimodalDataset(opt, mode)
                
                opt.dataset = "DHG"
                opt.full_modalities = [opt.modality, "depth"]
                dataset_2 = MultimodalDataset(opt, mode)
                
                opt.dataset = opt.original_dataset
                return dataset_1, dataset_2
            elif opt.exp_tag == "contrastive":
                if mode == "valid":
                    return None
                opt.full_modalities = ["skeleton", "stereo_ir", "depth"]
                prefix_DHG = f'./weights/{opt.exp_type}/DHG_pair_{opt.load_pretrain}_skeleton_{opt.seed}_{opt.dataset_split}_{opt.label_ratio}/paired_data_DHG_depth/'
                dataset_1 = MultimodalDataset(opt, mode, indice_prefix=prefix_DHG, indice_file=os.path.join(prefix_DHG, "train.txt"))
                
                prefix_GR4DHCI = f'./weights/{opt.exp_type}/GR4DHCI_pair_{opt.load_pretrain}_skeleton_{opt.seed}_{opt.dataset_split}_{opt.label_ratio}/paired_data_GR4DHCI_stereo_ir/'
                dataset_2 = MultimodalDataset(opt, mode, indice_prefix=prefix_GR4DHCI, indice_file=os.path.join(prefix_GR4DHCI, "train.txt"))
                return dataset_1, dataset_2
            else:
                raise NotImplementedError

def init_dataloader(opt, mode="train"):
    datasets = init_dataset(opt, mode)
    if datasets is None:
        return None
    if opt.exp_tag == "pair":
        dataset_1, dataset_2 = datasets
        val_datasets1, val_datasets2 = init_dataset(opt, "valid")
        dataset_1 = ConcatDataset([dataset_1, val_datasets1])
        dataset_2 = ConcatDataset([dataset_2, val_datasets2])
        return DataLoader(dataset_1, batch_size=opt.batch_size, shuffle=mode=="train"), DataLoader(dataset_2, batch_size=opt.batch_size, shuffle=mode=="train")
    elif opt.exp_tag == "contrastive":
        dataset_1, dataset_2 = datasets # GR4DHCI binded, DHG binded
        dataset = ConcatDataset([dataset_1, dataset_2])
        return DataLoader(dataset, batch_size=opt.batch_size, shuffle=mode=="train")
        
    else:
        return DataLoader(datasets, batch_size=opt.batch_size, shuffle=mode=="train")



MEAN = {
    "DHG": {
        "skeleton": [0.38020407, -0.27429489, 0.47039218],
        "depth": 77.56293159405392
    },
    "GR4DHCI": {
        # [  8.44332459 168.87676101 -13.76624872]
        "skeleton": [ 8.44332459, 168.87676101, -13.76624872],
        "stereo_ir": 13.26685933767596,
    },
    "Briareo": {
        "skeleton": [-13.1236488316878,365.27027607542004,-64.68353447994669],
        "stereo_ir": 5564.754052439263,
        "depth": 0.15057272356968843
    }
}

STD = {
    "DHG": {
        "skeleton": [0.13957358, 0.102889, 0.13977207],
        "depth": 223.12473617119093
    },
    "GR4DHCI": {
        "skeleton": [47.36835032, 105.23870595, 50.04857488],
        "stereo_ir": 24.403275576426545,
    },
    "Briareo": {
        "skeleton": [51.90136332536428,146.69681173481726,56.90887554328074],
        "stereo_ir": 11499.199537823684,
        "depth": 0.17632465308984457
    }
}

def normalize_dataset_mod(data, mod, dataset):
    if mod == "skeleton":
        data = data.reshape(30, -1, 3)
        for i in range(3):
            data[:, :, i] = (data[:, :, i] - MEAN[dataset][mod][i]) / STD[dataset][mod][i]
        data = data.reshape(30, -1)
    else:
        data = (data - MEAN[dataset][mod]) / STD[dataset][mod]
    return data
    
class MultimodalDataset():
    def __init__(self, opt, mode="train", indice_prefix=None, indice_file=None):
        self.opt = opt
        self.dataset = opt.dataset
        self.modalities = opt.full_modalities
        if indice_file is None:
            self.indice_file = f"./indices/{self.dataset}/{opt.dataset_split}"
            self.indice_file = os.path.join(self.indice_file, f"{mode}.txt")
            if not os.path.exists(self.indice_file):
                raise ValueError(f"{opt.indice_file} not found, please generate with preprocess.py/generate_index.py")
        else:
            self.indice_file = indice_file
        self.indices = np.loadtxt(self.indice_file, dtype=str)
        
        log.logprint(f"Loading {mode} dataset from {self.indice_file} for {self.modalities}")
        
        if indice_prefix is None:
            self.indice_prefix = os.path.join(opt.processed_data_path, self.dataset)
        else:
            self.indice_prefix = indice_prefix
        self.labels = np.load(os.path.join(self.indice_prefix, "label.npy"))
        
        if mode == 'train' and opt.label_ratio < 1.0:
            log.logprint(f"Using {opt.label_ratio} of the labels")
            # self.indices = self.indices[:int(len(self.indices) * opt.label_ratio)]
            label_indices = np.random.choice(len(self.indices), int(len(self.indices) * opt.label_ratio), replace=False)
            self.indices = self.indices[label_indices]
        
        log.logprint(f"Loading {mode} dataset from {self.indice_file} with {len(self.indices)} samples for {self.modalities}")
    
    def __len__(self):
        return len(self.indices)
    
    def __transform__(self, data):
        if self.opt.exp_tag != "contrastive":
            # Paired data are already normalized
            data = self.__normalize__(data)
        for mod in self.modalities:
            if mod == "stereo_ir" or mod == "depth":
                # add transform to resize from 30 * 171 * 224 to 30 * 112 * 112
                data[mod] = torch.nn.functional.interpolate(data[mod].unsqueeze(0), size=(112, 112), mode="bilinear", align_corners=False).squeeze(0)
        
        return data
        
    def __normalize__(self, data):
        for mod in self.modalities:
            data[mod] = normalize_dataset_mod(data[mod], mod, self.dataset)
        return data

    def __getitem__(self, idx):
        file_index = self.indices[idx]
        file_id = int(file_index)
        data = {}
        for mod in self.modalities:
            file_path = os.path.join(self.indice_prefix, mod, f"{file_index}.npy")
            mod_data = np.load(file_path).astype(np.float32)
            data[mod] = torch.from_numpy(mod_data)
        
        data = self.__transform__(data)
        label = self.labels[file_id]
        if isinstance(label, str):
            label = int(label)
        return data, label, file_id