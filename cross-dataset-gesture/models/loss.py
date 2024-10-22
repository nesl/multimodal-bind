import torch
import torch.nn as nn

def init_loss(opt):
    if opt.stage == "eval":
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError