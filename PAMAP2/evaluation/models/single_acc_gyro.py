import torch
import torch.nn as nn
# import numpy as np

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



class acc_decoder(nn.Module):
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
            nn.ConvTranspose2d(16, 32, 1),  # Inverse of previous Conv2d
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 64, 1),  # Inverse of previous Conv2d
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, input_size, 2),  # Inverse of previous Conv2d
            nn.BatchNorm2d(input_size),
            nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(120, 238, 2, batch_first=True)

    def forward(self, x):

        self.gru.flatten_parameters()

        x = x.view(x.size(0), 16, 120)

        x, _ = self.gru(x)

        x = x.view(x.size(0), 16, 119, 2)

        x = self.features(x)

        # print(x.shape)

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


class gyro_decoder(nn.Module):
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
            nn.ConvTranspose2d(16, 32, 1),  # Inverse of previous Conv2d
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 64, 1),  # Inverse of previous Conv2d
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, input_size, 2),  # Inverse of previous Conv2d
            nn.BatchNorm2d(input_size),
            nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(120, 238, 2, batch_first=True)

    def forward(self, x):

        self.gru.flatten_parameters()

        x = x.view(x.size(0), 16, 120)

        x, _ = self.gru(x)

        x = x.view(x.size(0), 16, 119, 2)

        x = self.features(x)

        # print(x.shape)

        return x



class MyUTDmodel_acc_AE(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size):
        super().__init__()

        self.acc_encoder = acc_encoder(input_size)
        self.acc_decoder = acc_decoder(input_size)


    def forward(self, x):

        output = self.acc_encoder(x)
        output = self.acc_decoder(output)

        return output


class MyUTDmodel_gyro_AE(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size):
        super().__init__()

        self.gyro_encoder = gyro_encoder(input_size)
        self.gyro_decoder = gyro_decoder(input_size)


    def forward(self, x):

        output = self.gyro_encoder(x)
        output = self.gyro_decoder(output)

        return output




