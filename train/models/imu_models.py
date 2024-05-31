import torch
import torch.nn as nn

class SingleIMUEncoder(nn.Module):
    def __init__(self, modality):
        super(SingleIMUEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 2),
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


        self.modality = modality
        self.GRU = nn.GRU(1998, hidden_size=120, num_layers=2, batch_first=True)

    def forward(self, data):
        self.GRU.flatten_parameters()
        acc_data = data[self.modality].cuda()
        batch_size = acc_data.shape[0]
        acc_data = torch.unsqueeze(acc_data, dim=1) # Add 1 channel
        embedding = torch.reshape(self.conv_layers(acc_data), (batch_size, 16, -1))
        embedding = self.GRU(embedding)[0]
        return embedding # batch_size x 16 * 120
    
class SingleIMUDecoder(nn.Module):
    def __init__(self):
        super(SingleIMUDecoder, self).__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 1),  # Inverse of previous Conv2d
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 64, 1),  # Inverse of previous Conv2d
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, 2),  # Inverse of previous Conv2d
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.GRU = nn.GRU(120, 1998, num_layers=2, batch_first=True)
        
        
    def forward(self, embedding):
        batch_size = embedding.shape[0]
        self.GRU.flatten_parameters()
        output = self.GRU(embedding)[0]
        output = torch.reshape(output, (batch_size, 16, 999, 2))
        output = self.deconv_layers(output)
        return output
    
class SingleIMUAutoencoder(nn.Module):
    def __init__(self, modality):
        super(SingleIMUAutoencoder, self).__init__()
        self.encoder = SingleIMUEncoder(modality)
        self.decoder = SingleIMUDecoder()
    def forward(self, data):
        embed = self.encoder(data)
        reconstructed = self.decoder(embed)
        return reconstructed


class FullIMUEncoder(nn.Module):
    def __init__(self):
        super(FullIMUEncoder, self).__init__()
        self.acc_encoder = SingleIMUEncoder('acc')
        self.gyro_encoder = SingleIMUEncoder('gyro')
        self.mag_encoder = SingleIMUEncoder('mag')
        self.acc_adapter = nn.Sequential(

            nn.Linear(1920, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),            
            )

        self.gyro_adapter= nn.Sequential(

            nn.Linear(1920, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),            
            )

        self.mag_adapter = nn.Sequential(

            nn.Linear(1920, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),            
            )

    def forward(self, batched_data):
        acc_embed = self.acc_encoder(batched_data)
        gyro_embed = self.gyro_encoder(batched_data)
        mag_embed = self.mag_encoder(batched_data)
        batch_size = acc_embed.shape[0]
        acc_embed = self.acc_adapter(torch.reshape(acc_embed, (batch_size, -1)))
        gyro_embed = self.gyro_adapter(torch.reshape(gyro_embed, (batch_size, -1)))
        mag_embed = self.mag_adapter(torch.reshape(mag_embed, (batch_size, -1)))

        return nn.functional.normalize(acc_embed), nn.functional.normalize(gyro_embed), nn.functional.normalize(mag_embed)
    

class GyroMagEncoder(nn.Module):
    def __init__(self):
        super(GyroMagEncoder, self).__init__()
        self.gyro_encoder = SingleIMUEncoder('gyro')
        self.mag_encoder = SingleIMUEncoder('mag')

        self.gyro_adapter= nn.Sequential(

            nn.Linear(1920, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),            
            )

        self.mag_adapter = nn.Sequential(

            nn.Linear(1920, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),            
            )

    def forward(self, batched_data):
        gyro_embed = self.gyro_encoder(batched_data)
        mag_embed = self.mag_encoder(batched_data)
        batch_size = gyro_embed.shape[0]
        gyro_embed = self.gyro_adapter(torch.reshape(gyro_embed, (batch_size, -1)))
        mag_embed = self.mag_adapter(torch.reshape(mag_embed, (batch_size, -1)))

        return nn.functional.normalize(gyro_embed), nn.functional.normalize(mag_embed)

class SupervisedGyroMag(nn.Module):
    def __init__(self):
        super(SupervisedGyroMag, self).__init__()
        self.gyro_encoder = SingleIMUEncoder('gyro')
        self.mag_encoder = SingleIMUEncoder('mag')
        self.output_head = nn.Sequential(
            nn.Linear(1920 * 2, 1920),
            nn.ReLU(),
            nn.Linear(1920, 7)
        )
    def forward(self, data):
        
        gyro_embed = self.gyro_encoder(data)
        
        b_size = gyro_embed.shape[0]
        gyro_embed = torch.reshape(gyro_embed, (b_size, -1))
        mag_embed = self.mag_encoder(data)
        mag_embed = torch.reshape(mag_embed, (b_size, -1))
        
        combined = torch.cat((gyro_embed, mag_embed), dim=-1)
        return self.output_head(combined)



if __name__ == '__main__':

    model = SingleIMUDecoder()
    model.cuda()
    embed = torch.randn(32, 928).cuda()
    print(model(embed).shape)