import torch
import torch.nn as nn
import torch.nn.functional as F

from models.single_modality import acc_encoder, gyro_encoder, skeleton_encoder
    

class MyUTDmodel_acc_gyro(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.acc_encoder = acc_encoder(input_size)
        self.gyro_encoder = gyro_encoder(input_size)

        # Classify output, fully connected layers
        self.classifier = nn.Sequential(

            nn.Linear(3840, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes),
            )


    def forward(self, x1, x2):

        acc_output = self.acc_encoder(x1)
        gyro_output = self.gyro_encoder(x2)

        fused_feature = torch.cat((acc_output,gyro_output), dim=1) #concate
        
        output = self.classifier(fused_feature)

        return output


class MyUTDmodel_skeleton_gyro(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.skeleton_encoder = skeleton_encoder(input_size)
        self.gyro_encoder = gyro_encoder(input_size)

        # Classify output, fully connected layers
        self.classifier = nn.Sequential(

            nn.Linear(3840, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes),
            )


    def forward(self, x1, x2):

        skeleton_output = self.skeleton_encoder(x1)
        gyro_output = self.gyro_encoder(x2)

        fused_feature = torch.cat((skeleton_output,gyro_output), dim=1) #concate
        
        output = self.classifier(fused_feature)

        return output
    
class MyUTDmodel_acc_skeleton(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.acc_encoder = acc_encoder(input_size)
        self.skeleton_encoder = skeleton_encoder(input_size)

        # Classify output, fully connected layers
        self.classifier = nn.Sequential(

            nn.Linear(3840, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes),
            )


    def forward(self, x1, x2):

        acc_output = self.acc_encoder(x1)
        skeleton_output = self.skeleton_encoder(x2)

        fused_feature = torch.cat((acc_output,skeleton_output), dim=1) #concate
        
        output = self.classifier(fused_feature)

        return output