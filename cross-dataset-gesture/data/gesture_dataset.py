import os
import numpy as np
import torch
import utils.log as log

class MultimodalDataset():
    def __init__(self, opt, mode="train"):
        self.opt = opt
        self.indice_file = os.path.join(opt.indice_file, f"{mode}.txt")
        self.modalities = ["skeleton", "stereo_ir", "depth"]

        self.indices = np.loadtxt(self.indice_file, dtype=str)
        self.indice_prefix = os.path.join(opt.processed_data_path, opt.dataset)
        self.labels = np.load(os.path.join(self.indice_prefix, "label.npy"))
        
        if mode == 'train' and opt.label_ratio < 1.0:
            log.logprint(f"Using {opt.label_ratio} of the labels")
            # self.indices = self.indices[:int(len(self.indices) * opt.label_ratio)]
            label_indices = np.random.choice(len(self.indices), int(len(self.indices) * opt.label_ratio), replace=False)
            self.indices = self.indices[label_indices]
        
        log.logprint(f"Loading dataset from {self.indice_file} with {len(self.indices)} samples")
    
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
        
        return data, label