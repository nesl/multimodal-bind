import torch.nn as nn
import utils.log as log
from models.model_utils import TransformerEnc

class SkeletonDecoder(nn.Module):
    def __init__(self):
        super(SkeletonDecoder, self).__init__()
        self.features_decoder = nn.Sequential(
            nn.ConvTranspose3d(16, 32, [3, 3, 1]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(32, 64, [3, 3, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(64, 64, [3, 3, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(64, 64, [3, 3, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 1, [5, 5, 2]),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.view(x.size(0), 16, 18, 9, 2)  # Reshape to match last layer of encoder
        x = self.features_decoder(x)
        x = x.squeeze(1)  # Remove the channel dimension
        x = x.view(x.size(0), 30, 63)  # Reshape to (b, 30, 63)
        return x

class SkeletonTransformerDecoder(nn.Module):
    def __init__(self):
        super(SkeletonTransformerDecoder, self).__init__()
        dim = 64
        log.logprint("Initializing Skeleton Transformer Decoder")
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 63),
        )
        self.decoder = TransformerEnc(dim=dim, depth=8, heads=4, dim_head=dim // 4, mlp_dim=dim * 4, dropout=0.1)

    def forward(self, x):
        b, dim = x.shape # (bsz, 30 * 64)
        x = x.view(b, 30, dim // 30)  # (bsz, 30, 64)
        x = self.decoder(x) # (bsz, 30, 64)
        x = x.reshape(b * 30, -1) # (bsz * 30, 64)
        x = self.adapter(x) # (bsz * 30, 63)
        x = x.reshape(b, 30, -1) # (bsz, 30, 64)
        return x

class StereoDecoder(nn.Module):
    def __init__(self):
        super(StereoDecoder, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class DepthDecoder(nn.Module):
    def __init__(self):
        super(DepthDecoder, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
