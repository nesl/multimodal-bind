import torch.nn as nn
import utils.log as log

class SkeletonDecoder(nn.Module):
    def __init__(self):
        super(SkeletonDecoder, self).__init__()
        self.gru = nn.GRU(120, 324, 2, batch_first=True)
        self.features = nn.Sequential(
            nn.ConvTranspose3d(16, 32, [3, 3, 1]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.ConvTranspose3d(32, 64, [3, 3, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.ConvTranspose3d(64, 64, [3, 3, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.ConvTranspose3d(64, 64, [3, 3, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.ConvTranspose3d(64, 1, [5, 5, 2]),
        )

    def forward(self, x):
        x, _ = self.gru(x)
        x = x.view(x.size(0), 16, 3, 30, 21)
        x = self.features(x)
        x = x.squeeze(1)
        return x


class StereoDecoder(nn.Module):
    def __init__(self):
        super(StereoDecoder, self).__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.ConvTranspose3d(32, 64, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.ConvTranspose3d(32, 1, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = x.view(x.size(0), 32, 1, 7, 7)  # Adjusting dimensions as per the original input size
        x = self.features(x)
        return x


class DepthDecoder(nn.Module):
    def __init__(self):
        super(DepthDecoder, self).__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.ConvTranspose3d(32, 64, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.ConvTranspose3d(32, 1, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = x.view(x.size(0), 32, 1, 7, 7)  # Adjusting dimensions as per the original input size
        x = self.features(x)
        return x
