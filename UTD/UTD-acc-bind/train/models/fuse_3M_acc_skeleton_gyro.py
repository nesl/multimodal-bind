import torch
import torch.nn as nn
import torch.nn.functional as F

from models.single_modality import acc_encoder, acc_decoder, gyro_encoder, gyro_decoder, skeleton_encoder, skeleton_decoder

    
class MyUTDmodel_3M_contrastive(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size):
        super().__init__()

        self.acc_encoder = acc_encoder(input_size)
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

        self.head_3 = nn.Sequential(

            nn.Linear(1920, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),            
            )
        

    def forward(self, x1, x2, x3, mask = None):

        acc_output = self.acc_encoder(x1)
        skeleton_output = self.skeleton_encoder(x2)
        gyro_output = self.gyro_encoder(x3)

        acc_feature_normalize = F.normalize(self.head_1(acc_output), dim=1)
        skeleton_feature_normalize = F.normalize(self.head_2(skeleton_output), dim=1)
        gyro_feature_normalize = F.normalize(self.head_3(gyro_output), dim=1)

        return acc_feature_normalize, skeleton_feature_normalize, gyro_feature_normalize


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

    def forward(self, x1, x2, x3, mask):

        acc_output = self.acc_encoder(x1)
        skeleton_output = self.skeleton_encoder(x2)
        gyro_output = self.gyro_encoder(x3)

        fused_feature = torch.cat((acc_output, skeleton_output, gyro_output), dim=1) #concate

        output = self.classifier(fused_feature)

        return output
    

    

class MyUTDmodel_3M_AE(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size):
        super().__init__()

        self.acc_encoder = acc_encoder(input_size)
        self.acc_decoder = acc_decoder(input_size)

        self.skeleton_encoder = skeleton_encoder(input_size)
        self.skeleton_decoder = skeleton_decoder(input_size)

        self.gyro_encoder = gyro_encoder(input_size)
        self.gyro_decoder = gyro_decoder(input_size)

        self.head = nn.Sequential(

            nn.Linear(5760, 1920),
            nn.BatchNorm1d(1920),
            nn.ReLU(inplace=True),
         
            )


    def forward(self, x1, x2, x3, mask):

        acc_feature = self.acc_encoder(x1)
        skeleton_feature = self.skeleton_encoder(x2)
        gyro_feature = self.gyro_encoder(x3)

        #print(acc_feature.shape, gyro_feature.shape)
        output = torch.cat((acc_feature,skeleton_feature, gyro_feature), dim=1) #concate
        output = self.head(output)

        acc_output = self.acc_decoder(output)
        skeleton_output = self.skeleton_decoder(output)
        gyro_output = self.gyro_decoder(output)

        return acc_output, skeleton_output, gyro_output
    
class MyUTDmodel_3M_contrastive_mask2(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size):
        super().__init__()

        self.acc_encoder = acc_encoder(input_size)
        self.skeleton_encoder = skeleton_encoder(input_size)
        self.gyro_encoder = gyro_encoder(input_size)

        self.head_1 = nn.Sequential(

            nn.Linear(1923, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),            
            )

        self.head_2 = nn.Sequential(

            nn.Linear(1923, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),            
            )

        self.head_3 = nn.Sequential(

            nn.Linear(1923, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),            
            )
        

    def forward(self, x1, x2, x3, mask):

        acc_output = self.acc_encoder(x1)
        skeleton_output = self.skeleton_encoder(x2)
        gyro_output = self.gyro_encoder(x3)

        # print(acc_output.shape)
        # print(mask.shape)
        acc_output = torch.cat((acc_output, mask.squeeze(1)), dim=1) #concate
        skeleton_output = torch.cat((skeleton_output, mask.squeeze(1)), dim=1) #concate
        gyro_output = torch.cat((gyro_output, mask.squeeze(1)), dim=1) #concate

        acc_feature_normalize = F.normalize(self.head_1(acc_output), dim=1)
        skeleton_feature_normalize = F.normalize(self.head_2(skeleton_output), dim=1)
        gyro_feature_normalize = F.normalize(self.head_3(gyro_output), dim=1)

        # print("after mask:", acc_feature_normalize[0].mean())

        return acc_feature_normalize, skeleton_feature_normalize, gyro_feature_normalize
    

