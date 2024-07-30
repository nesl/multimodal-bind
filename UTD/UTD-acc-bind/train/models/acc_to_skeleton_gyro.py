import torch
import torch.nn as nn
import torch.nn.functional as F

from models.single_modality import acc_encoder, gyro_decoder, skeleton_decoder
    

class MyUTDmodel_acc_to_skeleton(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size):
        super().__init__()

        self.acc_encoder = acc_encoder(input_size)
        self.skeleton_decoder = skeleton_decoder(input_size)
        
    def forward(self, x1):

        feature = self.acc_encoder(x1)

        output = self.skeleton_decoder(feature)


        return output
    

class MyUTDmodel_acc_to_gyro(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size):
        super().__init__()

        self.acc_encoder = acc_encoder(input_size)
        self.gyro_decoder = gyro_decoder(input_size)

        
    def forward(self, x1):

        feature = self.acc_encoder(x1)

        gyro_output = self.gyro_decoder(feature)


        return gyro_output

