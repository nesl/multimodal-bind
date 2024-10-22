import torch.nn as nn
import utils.log as log


INPUT_SIZE = {"skeleton": (30, 63), "stereo_ir": (30, 171, 224), "depth": (30, 171, 224)}


class SkeletonEncoder(nn.Module):
    def __init__(self):
        super(SkeletonEncoder, self).__init__()

        #  (40, 20, 3)
        #  (30, 21, 3)
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, [5, 5, 2]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv3d(64, 64, [3, 3, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Conv3d(64, 64, [3, 3, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Conv3d(64, 32, [3, 3, 1]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Conv3d(32, 16, [3, 3, 1]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(324, 120, 2, batch_first=True)

    def forward(self, x):
        b, t, c = x.size()
        x = x.view(b, t, c // 3, 3)
        x = x.unsqueeze(1)
        
        # print(x.size())
        x = self.features(x)
        # x = x.view(x.size(0), 16, -1)
        # x, _ = self.gru(x)
        x = x.reshape(x.size(0), -1)
        return x


class StereoEncoder(nn.Module):
    def __init__(self):
        super(StereoEncoder, self).__init__()
        input_size = INPUT_SIZE["stereo_ir"][1]
        log.logprint(f"Initializing Stereo IR Encoder with input shape: {input_size}")

        self.features = nn.Sequential(
            # First block: Conv3D -> BatchNorm -> ReLU -> MaxPool
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Second block: Conv3D -> BatchNorm -> ReLU -> MaxPool
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Third block: Conv3D -> BatchNorm -> ReLU -> MaxPool
            nn.Conv3d(64, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 30, 112, 112)
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return x


class DepthEncoder(nn.Module):
    def __init__(self):
        super(DepthEncoder, self).__init__()
        input_size = INPUT_SIZE["depth"][1]
        log.logprint(f"Initializing Depth Encoder with input shape: {input_size}")

        self.features = nn.Sequential(
            # First block: Conv3D -> BatchNorm -> ReLU -> MaxPool
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Second block: Conv3D -> BatchNorm -> ReLU -> MaxPool
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Third block: Conv3D -> BatchNorm -> ReLU -> MaxPool
            nn.Conv3d(64, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 30, 112, 112)
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return x
