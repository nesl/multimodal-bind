import torch
import torch.nn as nn
import utils.log as log

from models.model_utils import TransformerEnc


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

    def forward(self, x):
        b, t, c = x.size()
        x = x.view(b, t, c // 3, 3)
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return x


class SkeletonTransformerEncoder(nn.Module):
    def __init__(self):
        super(SkeletonTransformerEncoder, self).__init__()
        dim = 64
        self.tokenizer = nn.Sequential(
            nn.Linear(63, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        
        log.logprint("Initializing Skeleton Transformer Encoder")

        self.encoder = TransformerEnc(dim=dim, depth=8, heads=4, dim_head=dim // 4, mlp_dim=dim * 4, dropout=0.1)

        positions = torch.arange(0, 30).unsqueeze_(1)
        self.pos_embeddings = torch.zeros(30, dim).cuda()
        denominators = torch.pow(10000, 2 * torch.arange(0, dim // 2) / dim)  # 10000^(2i/d_model), i is the index of embedding
        self.pos_embeddings[:, 0::2] = torch.sin(positions / denominators)  # sin(pos/10000^(2i/d_model))
        self.pos_embeddings[:, 1::2] = torch.cos(positions / denominators)  # cos(pos/10000^(2i/d_model))

    def forward(self, x):
        b, t, c = x.shape # (bsz, 30, 63)
        x = x.view(b * t, c)  # (b, t, c) -> (b*t, c)
        x = self.tokenizer(x)
        x = x.reshape(b, t, -1)
        x = x + self.pos_embeddings
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
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
