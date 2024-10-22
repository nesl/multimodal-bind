import torch.nn as nn
import utils.log as log

class SkeletonDecoder(nn.Module):
    def __init__(self):
        super(SkeletonDecoder, self).__init__()
        
        # Inverse of the encoder's final pooling
        self.unflatten = nn.Unflatten(1, (512, 1, 1, 1))
        
        # Upsampling layers
        self.features = nn.Sequential(
            # Upsample Block 1
            nn.ConvTranspose3d(512, 512, kernel_size=(3, 3, 1), stride=(2, 2, 2), padding=(1, 1, 0), output_padding=(1, 1, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            # Upsample Block 2
            nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            # Upsample Block 3
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            # Upsample Block 4
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            # Final layer to match input channels
            nn.ConvTranspose3d(64, 1, kernel_size=(3, 3, 2), stride=(1, 1, 1), padding=(1, 1, 1))
        )
        
    def forward(self, x):
        x = self.unflatten(x)
        x = self.features(x)
        b = x.size(0)
        x = x.squeeze(1)
        x = x.view(b, 30, 63)  # Reshape to match original input size
        return x

class StereoDecoder(nn.Module):
    def __init__(self):
        super(StereoDecoder, self).__init__()
        
        self.unflatten = nn.Unflatten(1, (512, 1, 1, 1))
        
        self.features = nn.Sequential(
            # Upsample Block 1 (inverse of final pooling)
            nn.ConvTranspose3d(512, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Dropout(),
            
            # Upsample Block 2
            nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout(),
            
            # Upsample Block 3
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout(),
            
            # Upsample Block 4
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=(1, 1, 1), output_padding=(0, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout(),
            
            # Final Upsample Block
            nn.ConvTranspose3d(64, 1, kernel_size=(3, 7, 7), stride=(1, 3, 3), padding=(1, 1, 1), output_padding=(0, 2, 2))
        )
        
    def forward(self, x):
        x = self.unflatten(x)
        x = self.features(x)
        x = x.view(x.size(0), 30, 171, 224)  # Reshape to match original input size
        return x

class DepthDecoder(nn.Module):
    def __init__(self):
        super(DepthDecoder, self).__init__()
        
        self.unflatten = nn.Unflatten(1, (512, 1, 1, 1))
        
        self.features = nn.Sequential(
            # Upsample Block 1 (inverse of final pooling)
            nn.ConvTranspose3d(512, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Dropout(),
            
            # Upsample Block 2
            nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout(),
            
            # Upsample Block 3
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout(),
            
            # Upsample Block 4
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=(1, 1, 1), output_padding=(0, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout(),
            
            # Final Upsample Block
            nn.ConvTranspose3d(64, 1, kernel_size=(3, 7, 7), stride=(1, 3, 3), padding=(1, 1, 1), output_padding=(0, 2, 2))
        )
        
    def forward(self, x):
        x = self.unflatten(x)
        x = self.features(x)
        x = x.view(x.size(0), 30, 171, 224)  # Reshape to match original input size
        return x