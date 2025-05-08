import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x
    
class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x

    

class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim=32, h_dim=32, out_dim=3, n_res_layers=10, res_h_dim=128):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=5, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=kernel,
                               stride=stride, padding=1),
                            nn.ReLU(),
            nn.ConvTranspose2d(128, out_dim, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)
    


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim=3, h_dim=32, n_res_layers=10, res_h_dim=128):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, 100, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(100, 256, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=5, padding='same'),
            # nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=5, padding='same'),
            # nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=kernel,
                      stride=5, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers)

        )

    def forward(self, x):
        # The only time we are using encoder by itself to process batched data is for images
        if isinstance(x, dict):
            x = x['img']
        return self.conv_stack(x)

# Assume that they take a batch of data 
class ImageAE(nn.Module):
    def __init__(self):
        super(ImageAE, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()

    def forward(self, x):
        img_data = x['img']
        return self.dec(self.enc(img_data))
    
class DepthAE(nn.Module):
    def __init__(self):
        super(DepthAE, self).__init__()
        self.enc = Encoder(in_dim=1)
        self.dec = Decoder(out_dim=1)
    def forward(self, x):
        depth_data = x['depth']
        return self.dec(self.enc(depth_data))
    
class SemSegAE(nn.Module):
    def __init__(self):
        super(SemSegAE, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()
    def forward(self, x):
        semseg = x['semseg']
        return self.dec(self.enc(semseg))

class Incomplete_Contrastive_3M(nn.Module):
    def __init__(self):
        super(Incomplete_Contrastive_3M, self).__init__()
        self.image_enc = Encoder()
        self.depth_enc = Encoder(in_dim=1)
        self.semseg_enc = Encoder()
        self.img_head = nn.Sequential(
            nn.Linear(1536, 500),
            nn.ReLU(),
            nn.Linear(500, 128)
        )
        self.depth_head = nn.Sequential(
            nn.Linear(1536, 500),
            nn.ReLU(),
            nn.Linear(500, 128)
        )
        self.semseg_head = nn.Sequential(
            nn.Linear(1536, 500),
            nn.ReLU(),
            nn.Linear(500, 128)
        )

    def forward(self, x):
        img_embed = self.image_enc(x['img'])
        depth_embed = self.depth_enc(x['depth'])
        semseg_embed = self.semseg_enc(x['semseg'])
        bsz = img_embed.shape[0]
        img_embed = torch.reshape(img_embed, (bsz, -1))
        depth_embed = torch.reshape(depth_embed, (bsz, -1))
        semseg_embed = torch.reshape(semseg_embed, (bsz, -1))
        return {
            'img': F.normalize(self.img_head(img_embed), dim=-1), 
            'depth': F.normalize(self.depth_head(depth_embed), dim=-1),
            'semseg': F.normalize(self.semseg_head(semseg_embed), dim=-1)
        }
    
class Incomplete_Contrastive_2M(nn.Module):
    def __init__(self):
        super(Incomplete_Contrastive_2M, self).__init__()
        self.depth_enc = Encoder(in_dim=1)
        self.semseg_enc = Encoder()
        self.depth_head = nn.Sequential(
            nn.Linear(1536, 500),
            nn.ReLU(),
            nn.Linear(500, 128)
        )
        self.semseg_head = nn.Sequential(
            nn.Linear(1536, 500),
            nn.ReLU(),
            nn.Linear(500, 128)
        )

    def forward(self, x):
        depth_embed = self.depth_enc(x['depth'])
        semseg_embed = self.semseg_enc(x['semseg'])
        bsz = depth_embed.shape[0]
        depth_embed = torch.reshape(depth_embed, (bsz, -1))
        semseg_embed = torch.reshape(semseg_embed, (bsz, -1))
        return {
            'depth': F.normalize(self.depth_head(depth_embed), dim=-1),
            'semseg': F.normalize(self.semseg_head(semseg_embed), dim=-1)
        }
    
class Masked_Incomplete_Contrastive_3M(nn.Module):
    def __init__(self):
        super(Masked_Incomplete_Contrastive_3M, self).__init__()
        self.image_enc = Encoder()
        self.depth_enc = Encoder(in_dim=1)
        self.semseg_enc = Encoder()
        self.img_head = nn.Sequential(
            nn.Linear(1539, 500),
            nn.ReLU(),
            nn.Linear(500, 128)
        )
        self.depth_head = nn.Sequential(
            nn.Linear(1539, 500),
            nn.ReLU(),
            nn.Linear(500, 128)
        )
        self.semseg_head = nn.Sequential(
            nn.Linear(1539, 500),
            nn.ReLU(),
            nn.Linear(500, 128)
        )

    def forward(self, x):
        mask = torch.squeeze(x['mask'])
        img_embed = self.image_enc(x['img'])
        depth_embed = self.depth_enc(x['depth'])
        semseg_embed = self.semseg_enc(x['semseg'])
        bsz = img_embed.shape[0]
        img_embed = torch.reshape(img_embed, (bsz, -1))
        depth_embed = torch.reshape(depth_embed, (bsz, -1))
        semseg_embed = torch.reshape(semseg_embed, (bsz, -1))

        img_embed = torch.cat((img_embed, mask), dim=-1)
        depth_embed = torch.cat((depth_embed, mask), dim=-1)
        semseg_embed = torch.cat((semseg_embed, mask), dim=-1)
        return {
            'img': F.normalize(self.img_head(img_embed), dim=-1), 
            'depth': F.normalize(self.depth_head(depth_embed), dim=-1),
            'semseg': F.normalize(self.semseg_head(semseg_embed), dim=-1)
        }
    
class CrossModalGeneration(nn.Module):
    def __init__(self, input_mod_channels, out_mod_channels):
        super(CrossModalGeneration, self).__init__()
        self.enc = Encoder(in_dim=input_mod_channels)
        self.dec = Decoder(out_dim=out_mod_channels)
    def forward(self, x): # always going to be image -> something
        return self.dec(self.enc(x['img']))

class SupervisedDepthSemseg(nn.Module):
    def __init__(self):
        super(SupervisedDepthSemseg, self).__init__()
        self.semseg_encoder = Encoder(in_dim = 3)
        self.depth_encoder = Encoder(in_dim=1)
        self.output_head = nn.Sequential(
            nn.Linear(1536 * 2, 500),
            nn.ReLU(),
            nn.Linear(500, 6)
        )
    def forward(self, x):
        semseg_embed = self.semseg_encoder(x['semseg'])
        depth_embed = self.depth_encoder(x['depth'])
        bsz = semseg_embed.shape[0]
        semseg_embed = torch.reshape(semseg_embed, (bsz, -1))
        depth_embed = torch.reshape(depth_embed, (bsz, -1))
        joint_embed = torch.cat((semseg_embed, depth_embed), dim=-1)
        return self.output_head(joint_embed)