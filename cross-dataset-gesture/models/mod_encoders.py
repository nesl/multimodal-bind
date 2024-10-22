import torch.nn as nn
import utils.log as log


INPUT_SIZE = {"skeleton": (30, 63), "stereo_ir": (30, 171, 224), "depth": (30, 171, 224)}


class SkeletonEncoder(nn.Module):
    def __init__(self):
        super(SkeletonEncoder, self).__init__()
        
        # 30, 21, 3

        # Extract features, 3D conv layers
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv3d(1, 64, kernel_size=(3, 3, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(1, 1, 1)),

            # # Conv Block 2
            nn.Conv3d(64, 128, kernel_size=(3, 3, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            # # Conv Block 3
            nn.Conv3d(128, 256, kernel_size=(3, 3, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            # # Conv Block 4
            nn.Conv3d(256, 512, kernel_size=(3, 3, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Conv3d(512, 512, kernel_size=(3, 3, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 2)),

            # Global Avg Pooling
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.gru = nn.GRU(162, 120, 2, batch_first=True)

    def forward(self, x):
        b, t, c = x.size()
        x = x.view(b, t, c // 3, 3)

        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.reshape(x.size(0), -1)

        return x


class StereoEncoder(nn.Module):
    def __init__(self):
        super(StereoEncoder, self).__init__()
        input_size = INPUT_SIZE["stereo_ir"][1]
        log.logprint(f"Initializing Stereo IR Encoder with input shape: {input_size}")
        # Extract features, 3D conv layers

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv3d(1, 64, kernel_size=(3, 7, 7), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=(1, 5, 5), stride=(1, 3, 3)),

            # Conv Block 2
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=(1, 5, 5), stride=(1, 3, 3)),

            # Conv Block 3
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            # Conv Block 4
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            # Conv Block 5
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)),

            # Global Average Pooling
            nn.AdaptiveAvgPool3d((1, 1, 1))
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

        # Extract features, 3D conv layers from input (112, 1122)
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv3d(1, 64, kernel_size=(3, 7, 7), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=(1, 5, 5), stride=(1, 3, 3)),

            # Conv Block 2
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=(1, 5, 5), stride=(1, 3, 3)),

            # Conv Block 3
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            # Conv Block 4
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            # Conv Block 5
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)),

            # Global Average Pooling
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
                          
    def forward(self, x):
        x = x.view(x.size(0), 1, 30, 112, 112)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
