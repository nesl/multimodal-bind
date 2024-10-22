import torch
import torch.nn as nn

def init_mmbind_loss(opt):
    if opt.exp_tag == "unimod":
        return nn.MSELoss()
    else:
        raise NotImplementedError
def init_loss(opt):
    if opt.stage == "eval":
        return nn.CrossEntropyLoss()
    else:
        if opt.exp_type == "mmbind":
            return init_mmbind_loss(opt)
        raise NotImplementedError