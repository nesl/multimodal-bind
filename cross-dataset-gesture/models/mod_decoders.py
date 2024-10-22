import torch.nn as nn
import utils.log as log

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
