import torch.nn as nn

class SkeletonEncoder(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, mod_data):
        """
        mod_data: torch.Tensor
            -- shape: (batch, 30, 64)
        """
        pass


class StereoEncoder(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, mod_data):
        """
        mod_data: torch.Tensor
            -- shape: (batch, 30, 171, 224)
        """
        pass


class DepthEncoder(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, mod_data):
        """
        mod_data: torch.Tensor
            -- shape: (batch, 30, 171, 224)
        """
        pass
