import torch
import torch.nn as nn


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



class skeleton_decoder(nn.Module):
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
        

        self.gru = nn.GRU(120, 448, 2, batch_first=True)


        # Extract features, 3D conv layers
        self.features = nn.Sequential(
            nn.ConvTranspose3d(16, 32, [3,3,1]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 64, [3,3,1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.ConvTranspose3d(64, 64, [3,3,1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.ConvTranspose3d(64, 64, [3,3,1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.ConvTranspose3d(64, 64, [3,3,1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.ConvTranspose3d(64, input_size, [3,3,2]),  # Adjust the kernel size to match the input size
            nn.BatchNorm3d(input_size),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):

        self.gru.flatten_parameters()

        x = x.view(x.size(0), 16, 120)

        x, _ = self.gru(x)

        x = x.view(x.size(0), 16, 28, 8, 2)

        # print(x.shape)

        x = self.features(x)

        # print(x.shape)

        return x



class MyUTDmodel_skeleton_AE(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size):
        super().__init__()

        self.skeleton_encoder = skeleton_encoder(input_size)
        self.skeleton_decoder = skeleton_decoder(input_size)


    def forward(self, x):

        output = self.skeleton_encoder(x)
        output = self.skeleton_decoder(output)

        return output


class MyUTDmodel_skeleton_encoder(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size):
        super().__init__()

        self.skeleton_encoder = skeleton_encoder(input_size)


    def forward(self, x):

        output = self.skeleton_encoder(x)

        return output

