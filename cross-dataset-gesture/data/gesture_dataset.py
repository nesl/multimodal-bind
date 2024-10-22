import os
import numpy as np
import torch
import utils.log as log
from torch.utils.data import ConcatDataset

def init_dataset(opt, mode="train"):
    opt.full_modalities = ["skeleton", "stereo_ir", "depth"]
    if opt.stage == "eval":
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
            else:
                raise NotImplementedError

class MultimodalDataset():
    def __init__(self, opt, mode="train"):
        self.opt = opt
        self.indice_file = f"./indices/{opt.dataset}/{opt.dataset_split}"
        self.indice_file = os.path.join(self.indice_file, f"{mode}.txt")
        if not os.path.exists(self.indice_file):
            raise ValueError(f"{opt.indice_file} not found, please generate with preprocess.py/generate_index.py")

        self.modalities = opt.full_modalities
        
        
        self.indices = np.loadtxt(self.indice_file, dtype=str)
        self.indice_prefix = os.path.join(opt.processed_data_path, opt.dataset)
        self.labels = np.load(os.path.join(self.indice_prefix, "label.npy"))
        
        if mode == 'train' and opt.label_ratio < 1.0:
            log.logprint(f"Using {opt.label_ratio} of the labels")
            # self.indices = self.indices[:int(len(self.indices) * opt.label_ratio)]
            label_indices = np.random.choice(len(self.indices), int(len(self.indices) * opt.label_ratio), replace=False)
            self.indices = self.indices[label_indices]
        
        log.logprint(f"Loading dataset from {self.indice_file} with {len(self.indices)} samples for {self.modalities}")
    
    def __len__(self):
        return len(self.indices)
    
    def __transform__(self, data):
        for mod in self.modalities:
            if mod == "stereo_ir" or mod == "depth":
                # add transform to resize from 30 * 171 * 224 to 30 * 112 * 112
                data[mod] = torch.nn.functional.interpolate(data[mod].unsqueeze(0), size=(112, 112), mode="bilinear", align_corners=False).squeeze(0)
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
        return data, label