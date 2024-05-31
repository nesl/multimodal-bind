import torch
import torch.nn as nn
import torch.nn.functional as F

class acc_encoder(nn.Module):
    """
    CNN layers applied on acc sensor data to generate pre-softmax
    ---
    params for __init__():
        input_size: e.g. 1
        num_classes: e.g. 6
    forward():
        Input: data
        Output: pre-softmax
    """
    def __init__(self, input_size):
        super().__init__()

        # Extract features, 2D conv layers
        self.features = nn.Sequential(
            nn.Conv2d(input_size, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            # nn.Conv2d(64, 64, 2),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),

            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            )

        self.gru = nn.GRU(238, 120, 2, batch_first=True)

    def forward(self, x):

        self.gru.flatten_parameters()
        x = self.features(x)

        x = x.view(x.size(0), 16, -1)
        # print(x.shape)

        x, _ = self.gru(x)

        x = x.reshape(x.size(0), -1)

        return x


class gyro_encoder(nn.Module):
    """
    CNN layers applied on acc sensor data to generate pre-softmax
    ---
    params for __init__():
        input_size: e.g. 1
        num_classes: e.g. 6
    forward():
        Input: data
        Output: pre-softmax
    """
    def __init__(self, input_size):
        super().__init__()

        # Extract features, 2D conv layers
        self.features = nn.Sequential(
            nn.Conv2d(input_size, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            # nn.Conv2d(64, 64, 2),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),

            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            )

        self.gru = nn.GRU(238, 120, 2, batch_first=True)

    def forward(self, x):

        self.gru.flatten_parameters()
        x = self.features(x)

        x = x.view(x.size(0), 16, -1)
        # print(x.shape)

        x, _ = self.gru(x)

        x = x.reshape(x.size(0), -1)

        return x


class skeleton_encoder(nn.Module):
    """
    CNN layers applied on skeleton sensor data to generate pre-softmax
    ---
    params for __init__():
        input_size: e.g. 1
        num_classes: e.g. 6
    forward():
        Input: data
        Output: pre-softmax
    """
    def __init__(self, input_size):
        super().__init__()

        # Extract features, 3D conv layers
        self.features = nn.Sequential(
            nn.Conv3d(input_size, 64, [5,5,2]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv3d(64, 64, [3,3,1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),


            nn.Conv3d(64, 64, [3,3,1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv3d(64, 32, [3,3,1]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv3d(32, 16, [3,3,1]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            )
        

        self.gru = nn.GRU(448, 120, 2, batch_first=True)




    def forward(self, x):

        self.gru.flatten_parameters()

        # print(x.shape)

        x = self.features(x)
        
        x = x.view(x.size(0), 16, -1)

        x, _ = self.gru(x)

        # x = x.contiguous().view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)

        # print(x.shape)

        return x
    
class MyUTDmodel_3M(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes):
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

    def forward(self, x1, x2, x3, mask=""):

        acc_output = self.acc_encoder(x1)
        skeleton_output = self.skeleton_encoder(x2)
        gyro_output = self.gyro_encoder(x3)

        acc_feature_normalize = F.normalize(self.head_1(acc_output), dim=1)
        skeleton_feature_normalize = F.normalize(self.head_2(skeleton_output), dim=1)
        gyro_feature_normalize = F.normalize(self.head_3(gyro_output), dim=1)

        return acc_feature_normalize, skeleton_feature_normalize, gyro_feature_normalize


class MyUTDmodel_3M_All(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes):
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