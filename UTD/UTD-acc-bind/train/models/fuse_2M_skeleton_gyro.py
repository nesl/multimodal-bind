import torch
import torch.nn as nn
import torch.nn.functional as F

from models.single_modality import skeleton_encoder, skeleton_decoder, gyro_encoder, gyro_decoder
    
class MyUTDmodel_2M_AE(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size):
        super().__init__()

        self.skeleton_encoder = skeleton_encoder(input_size)
        self.skeleton_decoder = skeleton_decoder(input_size)

        self.gyro_encoder = gyro_encoder(input_size)
        self.gyro_decoder = gyro_decoder(input_size)

        # Classify output, fully connected layers
        self.head = nn.Sequential(

            nn.Linear(3840, 1920),
            nn.BatchNorm1d(1920),
            nn.ReLU(inplace=True),

            # nn.Linear(1920, 1920),
            )
        
    def forward(self, x1, x2):

        skeleton_feature = self.skeleton_encoder(x1)
        gyro_feature = self.gyro_encoder(x2)

        #print(skeleton_feature.shape, gyro_feature.shape)
        output = torch.cat((skeleton_feature,gyro_feature), dim=1) #concate
        output = self.head(output)

        skeleton_output = self.skeleton_decoder(output)
        gyro_output = self.gyro_decoder(output)

        return skeleton_output, gyro_output
    


class MyUTDmodel_2M_contrastive(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size):
        super().__init__()

        self.skeleton_encoder = skeleton_encoder(input_size)
        self.gyro_encoder = gyro_encoder(input_size)

        self.head_1 = nn.Sequential(

            nn.Linear(1920, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),            
            )

        self.head_2 = nn.Sequential(

            nn.Linear(1920, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),            
            )

    def forward(self, x1, x2):

        skeleton_output = self.skeleton_encoder(x1)
        gyro_output = self.gyro_encoder(x2)

        skeleton_feature_normalize = F.normalize(self.head_1(skeleton_output), dim=1)
        gyro_feature_normalize = F.normalize(self.head_2(gyro_output), dim=1)

        return skeleton_feature_normalize, gyro_feature_normalize


class MyUTDmodel_2M(nn.Module):
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
