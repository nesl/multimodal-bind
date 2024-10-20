import numpy as np

class MultimodalDataset():
    def __init__(self, valid_actions, valid_mods, root="data/", opt=None):
        self.valid_actions = valid_actions
        self.valid_mods = valid_mods
        self.root = root
        
        index_file = ""
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass