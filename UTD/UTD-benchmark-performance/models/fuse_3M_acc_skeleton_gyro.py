import torch
import torch.nn as nn
import torch.nn.functional as F

from models.single_modality import acc_encoder, acc_decoder, gyro_encoder, gyro_decoder, skeleton_encoder, skeleton_decoder



class MyUTDmodel_3M(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.acc_encoder = acc_encoder(input_size)
        self.skeleton_encoder = skeleton_encoder(input_size)
        self.gyro_encoder = gyro_encoder(input_size)
        
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(

            nn.Linear(5760, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes),
            )

    def forward(self, x1, x2, x3):

        acc_output = self.acc_encoder(x1)
        skeleton_output = self.skeleton_encoder(x2)
        gyro_output = self.gyro_encoder(x3)

        fused_feature = torch.cat((acc_output, skeleton_output, gyro_output), dim=1) #concate

        output = self.classifier(fused_feature)

        return output
    

