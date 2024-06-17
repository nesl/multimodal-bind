from models.vit_dev import TransformerEnc, TransformerDec

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from einops import repeat
import numpy as np

# IDENTICAL TO model.py IN TRAIN

# Standard depth encoder
    # Output b_size x 1920
class DepthEncoder(nn.Module):
    def __init__(self):
        super(DepthEncoder, self).__init__()
        # Define CNN model, we have grayscale depth images of 1 x 48 x 64
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, (7, 7)),
            nn.Dropout(0.1),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (5, 5)),
            nn.Dropout(0.1),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (5, 5)),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3)),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (2, 2)),
            nn.Dropout(0.1),
        )
        d = 64
        # Project to smaller dimension before performing self-attention along the temporal dimension
        self.project_lin = nn.Linear(160, d)
        
        self.time_encoder = TransformerEnc(dim=d, depth=2, heads=2, dim_head=d//2, mlp_dim=3*d)
        
        # Define position embeddings for transformer, we process 30 frames
        positions = torch.arange(0, 30).unsqueeze_(1)
        self.pos_embeddings = torch.zeros(30, d)
        denominators = torch.pow(10000, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
        self.pos_embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
        self.pos_embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    # The model should receive the batched data only out of dictionary form
    # depth data has form batch_size * num_frames * 3 * 48 x 64 (downsampled by getitem in dataset)
    def forward(self, data, codebook):
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_embeddings = self.pos_embeddings.to(device)
        data = data['input_depth'][:, 0:30].to(device)

        batch_size, n_frames, channels, dim1, dim2 = data.shape

        # Merge the number of frames and batches before undergoing the conv network
        data = torch.reshape(data, (-1, channels, dim1, dim2))
        # Reshape output to batch_size x num_frames x 128
        resnet_output = torch.reshape(torch.squeeze(self.conv_layers(data)), (batch_size, n_frames, -1))
        # Project to 64
        data = self.project_lin(resnet_output)
        # Add the positional embeddings, and place into transformer encoder
        data += self.pos_embeddings[0:n_frames]
        data = self.time_encoder(data)
        # Return embedding in shape batch_size x 1920, note that normalization is done by the contrastive model
        output = torch.reshape(data, (batch_size, -1))
        similarity = nn.functional.normalize(torch.matmul(output, torch.transpose(codebook, 0, 1)))
        weights = torch.softmax(similarity, dim=-1) # 64 x 50
        output = torch.matmul(weights, codebook)
        return output 
    

# Reconstructs the original depth image
# Original image size is 3 x 48 x 64 (we have 30 of them)
class DepthReconstruct(nn.Module):
 
    def __init__(self):
        super(DepthReconstruct, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Arbitrarily set the token size to be 512
        dim=64
        self.project_lin = nn.Linear(dim, 128)
        # The idea is that we need to reconstruct 30 images of 3 x 48 x 64
        # This means we need (30 * 18) tokens of size 512 to get 276480 values, unsure if this is good
        #self.queries = nn.Parameter(torch.randn(30 * 6, dim))
        self.mask_token = nn.Parameter(torch.randn(dim))
        # Decoder takes our encoded data as input, and uses it to write data to the queries
        self.encoder = TransformerEnc(dim=dim, depth=6, heads=2, dim_head=dim//2, mlp_dim=dim*3)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 16, (7, 7), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, (7, 7), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, (4, 4), stride=(2, 2))

        )


    # Input is b_size x 1920 (30 x 64)
    def forward(self, depth_tokens):

        batch_size = depth_tokens.shape[0]
        depth_tokens = torch.reshape(depth_tokens, (batch_size, 30, 64))
        output = self.encoder(depth_tokens)
        output = self.project_lin(output)
        output = torch.reshape(output, (batch_size * 30, 16, 2, 4))
        output = self.deconv(output)
        return torch.reshape(output, (batch_size, 30, 48, 64))


        

class SkeletonEncoder(nn.Module):
    def __init__(self):
        super(SkeletonEncoder, self).__init__()
        d = 16
        # The purpose of the tokenizer is to reduce the input size, output of encoder must be smaller than input to enable reconstruction
        self.tokenizer = nn.Sequential(
            nn.Linear(34, 20),
            nn.ReLU(),
            nn.Linear(20, 16)
        )
        # Treat each data frame as one token as input to a transformer encoder
        self.encoder = TransformerEnc(dim=d, depth=12, heads=4, dim_head=d//3, mlp_dim=3*d) # Previously 8
        # Positional embeddings
        positions = torch.arange(0, 30).unsqueeze_(1)
        self.pos_embeddings = torch.zeros(30, d)
        denominators = torch.pow(10000, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
        self.pos_embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
        self.pos_embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    # Skeleton data (input_rgb) is shape b_size x 30 x 17 x 2
    def forward(self, data, codebook):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_embeddings = self.pos_embeddings.to(device)
        data = data['input_rgb'].to(device)
        batch_size, n_tokens, dim1, dim2 = data.shape
        # Pass through tokenizer after reshaping, not sure is reshape is necessary
        data = torch.reshape(data, (batch_size * n_tokens, -1))
        data = self.tokenizer(data)
        data = torch.reshape(data, (batch_size, n_tokens, -1))
        # Add pos embeddings, and return output of the encoder
        data += self.pos_embeddings
        data = self.encoder(data) # batch_size x 30 x 16
        output = torch.reshape(data,(batch_size, -1))
        similarity = nn.functional.normalize(torch.matmul(output, torch.transpose(codebook, 0, 1)))
        weights = torch.softmax(similarity, dim=-1) # 64 x 50
        output = torch.matmul(weights, codebook)
        return output # batch_size x 480

# Reconstruct the skeleton
class SkeletonReconstruct(nn.Module):
    def __init__(self):
        super(SkeletonReconstruct, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Choose token size of 34 since that corresponds to one frame
        dim=34
        # 30 tokens to reconstruct
        self.queries = nn.Parameter(torch.randn(30, dim))
        self.decoder = TransformerDec(dim=dim, depth=100, heads=4, dim_head=dim//4, mlp_dim=dim*3) # Was 6
        self.adapter = nn.Sequential(
            nn.Linear(34, 34)
        )
    # skeleton encoder output is batch_size x 480
    def forward(self, rgb_tokens):
        batch_size = rgb_tokens.shape[0]
        # Extract 14 tokens of size 34 (476) from the 480 length data (last four numbers do not contribute)
        rgb_tokens = rgb_tokens[:, :476] # 14 tokens of 34
        rgb_tokens = torch.reshape(rgb_tokens, (batch_size, -1, 34))

        queries = repeat(self.queries, 'n d -> b n d', b=batch_size)
        output_tokens = self.decoder(queries, rgb_tokens) 
        # Unsure if this reshape necessary
        reconstructed_tokens = torch.reshape(output_tokens, (-1, 34))
        transformed_tokens = self.adapter(reconstructed_tokens)
        return torch.reshape(transformed_tokens, (batch_size, -1, 34))

# mmWave is complicated, 30 x ? x 5, where we have varying number of points per frame
# Dataloader will zero pad to the largest number of points per frame
class mmWaveEncoder(nn.Module):

    def __init__(self):
        super(mmWaveEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Act as a query extracting information from varying size data, treat each point as a token
        self.keypoints = nn.Parameter(torch.randn((17, 5)))
        self.space_encoder = TransformerDec(dim=5, depth=6, heads=1, dim_head=5, mlp_dim=15)
        # Time dimension of 64 to match with Depth
        time_dim=64
        self.projector = nn.Linear(85, 64)
        # Positional Embeddings
        positions = torch.arange(0, 10000).unsqueeze_(1)
        self.pos_embeddings = torch.zeros(10000, time_dim)
        denominators = torch.pow(10000, 2*torch.arange(0, time_dim//2 + 1)/time_dim) # 10000^(2i/d_model), i is the index of embedding
        self.pos_embeddings[:, 0::2] = torch.sin(positions/denominators)[:, 0:self.pos_embeddings[:, 0::2].shape[-1]] # sin(pos/10000^(2i/d_model))
        self.pos_embeddings[:, 1::2] = torch.cos(positions/denominators)[:, 0:self.pos_embeddings[:, 1::2].shape[-1]] # cos(pos/10000^(2i/d_model))

        self.time_encoder = TransformerEnc(dim=time_dim, depth=4, heads=4, dim_head=time_dim//4, mlp_dim=time_dim*3)
    
    def forward(self, data, codebook):
        # Get only the first 30 frames of mmWave data (297 frames present)
        data = data['input_mmwave'][:, 0:30].to(self.device)
        batch_size, num_frames, num_pts, _ = data.shape
        # We apply space encoder on every frame, merge n_frames and batch_size
        data = torch.squeeze(torch.reshape(data, (batch_size * num_frames, num_pts, _)))
        queries = repeat(self.keypoints, 'n d -> b n d', b=batch_size*num_frames)
        space_embeddings = self.space_encoder(queries, data) # Extract the main 17 x 5 keypoints
        space_embeddings = torch.reshape(space_embeddings, (batch_size, num_frames, -1))
        space_embeddings = self.projector(space_embeddings)
        # Space embeddings are extracted from each frame of the data
        # Add positional embeddings and place into time encoder
        pos_embeddings = self.pos_embeddings[0:space_embeddings.shape[1]].to(torch.device('cuda'))
        pos_embeddings = repeat(pos_embeddings, 'n d -> b n d', b=batch_size)
        space_embeddings += pos_embeddings
        output_embeddings = self.time_encoder(space_embeddings)
        output = torch.reshape(output_embeddings, (batch_size, -1))
        similarity = nn.functional.normalize(torch.matmul(output, torch.transpose(codebook, 0, 1)))
        weights = torch.softmax(similarity, dim=-1) # 64 x 50
        output = torch.matmul(weights, codebook)
        return output



class mmWaveReconstruct(nn.Module):
    def __init__(self):
        super(mmWaveReconstruct, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Arbitrarily set the max number of reconstructed points to be 100 (if larger than 100 we ignore)
        # Thus, dimension of 500 corresponds to 100 points of size 5
        dim = 500 # (100 points)
        # Queries represent the target we want to reconstruct since model output is the number of queries
        # Thus we have 30 x 500 queries to represent 30 frames of 100 points of size 5
        self.queries = nn.Parameter(torch.randn(30, dim))
        self.decoder = TransformerDec(dim=dim, depth=4, heads=4, dim_head=dim//4, mlp_dim=dim*4)
        # Same size linear projection is to ensure model can control the scale of the input, using straight output of TD can have scale issues since transformer
        # cannot easily scale the values
        self.projector = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500)
        )
    # input is batch_size x 1920, 30 tokens
    def forward(self, mmwave_token):
        batch_size = mmwave_token.shape[0]
        # Take 3 tokens of size 500 from the encoder
        mmwave_token = mmwave_token[:, :1500]
        mmwave_token = torch.reshape(mmwave_token, (batch_size, -1, 500))

        queries = repeat(self.queries, 'n d -> b n d', b=batch_size)
        output_tokens = self.decoder(queries, mmwave_token) # b x 30 x 500
        reconstructed_tokens = self.projector(output_tokens)
        return reconstructed_tokens
    
# Skeleton autoencoder model utilized in mmbind_1 to perform pretraining
class SkeletonAE(nn.Module):
    def __init__(self):
        super(SkeletonAE, self).__init__()
        self.skeleton_encoder = SkeletonEncoder()
        self.skeleton_decoder = SkeletonReconstruct()
        self.codebook = nn.Parameter(torch.randn(50, 480))
    def forward(self, data):
        encoded = self.skeleton_encoder(data, self.codebook)
        return self.skeleton_decoder(encoded)

# Used in mmbind_3 to perform contrastive on the paired data
class mmWaveDepthContrastive(nn.Module):
    def __init__(self):
        super(mmWaveDepthContrastive, self).__init__()
        self.mmWave_encoder = mmWaveEncoder()
        self.depth_encoder = DepthEncoder()
        # Adapter to smaller size for contrastive learning
        self.mmWave_adapter = nn.Sequential(
            nn.Linear(1920, 800),
            nn.ReLU(),
            nn.Linear(800, 128)
        )
        self.depth_adapter = nn.Sequential(
            nn.Linear(1920, 800),
            nn.ReLU(),
            nn.Linear(800, 128)
        )
    def forward(self, data):
        # Provides the 128 embeddings for mmWave and depth
        mmWave_output = self.mmWave_adapter(self.mmWave_encoder(data)) 
        depth_output = self.depth_adapter(self.depth_encoder(data))
        # Normalize and return the embeddings
        return nn.functional.normalize(depth_output), nn.functional.normalize(mmWave_output)

# Utilized to train the DualContrastive Learning model
class DualContrastiveModel(nn.Module):
    def __init__(self):
        super(DualContrastiveModel, self).__init__()
        self.mmWave_encoder = mmWaveEncoder()
        self.depth_encoder = DepthEncoder()
        self.skeleton_encoder = SkeletonEncoder()
        self.mmWave_adapter = nn.Sequential(
            nn.Linear(1920, 800), # used to be 240 maybe that was why it does better?
            nn.ReLU(),
            nn.Linear(800, 128)
        )
        self.depth_adapter = nn.Sequential(
            nn.Linear(1920, 800),
            nn.ReLU(),
            nn.Linear(800, 128)
        )
        self.skeleton_adapter = nn.Sequential(
            nn.Linear(480, 240),
            nn.ReLU(),
            nn.Linear(240, 128)
        )
        self.codebook = nn.Parameter(torch.randn(5, 1920))
    def forward(self, data):
        mmWave_output= self.mmWave_adapter(self.mmWave_encoder(data, self.codebook))
        depth_output = self.depth_adapter(self.depth_encoder(data, self.codebook))
        skeleton_output = self.skeleton_adapter(self.skeleton_encoder(data, self.codebook[:, :480]))
        return nn.functional.normalize(depth_output), nn.functional.normalize(mmWave_output),\
            nn.functional.normalize(skeleton_output)



# Utilized to train the DualContrastive Learning model
class DualContrastiveModelOld(nn.Module):
    def __init__(self):
        super(DualContrastiveModelOld, self).__init__()
        self.mmWave_encoder = mmWaveEncoder()
        self.depth_encoder = DepthEncoder()
        self.skeleton_encoder = SkeletonEncoder()
        self.mmWave_adapter = nn.Sequential(
            nn.Linear(1920, 800), # used to be 240 maybe that was why it does better?
            nn.ReLU(),
            nn.Linear(800, 128)
        )
        self.depth_adapter = nn.Sequential(
            nn.Linear(1920, 800),
            nn.ReLU(),
            nn.Linear(800, 128)
        )
        self.skeleton_adapter = nn.Sequential(
            nn.Linear(480, 240),
            nn.ReLU(),
            nn.Linear(240, 128)
        )
    def forward(self, data):
        mmWave_output= self.mmWave_adapter(self.mmWave_encoder(data))
        depth_output = self.depth_adapter(self.depth_encoder(data))
        skeleton_output = self.skeleton_adapter(self.skeleton_encoder(data))
        return nn.functional.normalize(depth_output), nn.functional.normalize(mmWave_output),\
            nn.functional.normalize(skeleton_output)


# Supervised model, we will load the encoders if using pretrained
class mmWaveDepthSupervised(nn.Module):
    def __init__(self):
        super(mmWaveDepthSupervised, self).__init__()
        self.mmWave_encoder = mmWaveEncoder()
        self.depth_encoder = DepthEncoder()
        self.output_head = nn.Sequential(
            nn.Linear(64, 27)
        )
        # Use a transformer encoder to combine features
        self.combine_features = TransformerEnc(dim=64, depth=4, heads=4, dim_head=16, mlp_dim=64*3)
        self.cls = nn.Parameter(torch.randn((1, 1, 64)))

    def forward(self, data):
        
        mmWave_output = self.mmWave_encoder(data)
        depth_output = self.depth_encoder(data)
        batch_size = depth_output.shape[0]
        # 30 tokens of size 64 for both
        mmWave_output = torch.reshape(mmWave_output, (batch_size, -1, 64))
        depth_output = torch.reshape(depth_output, (batch_size, -1, 64))

        cls = repeat(self.cls, '1 1 n -> b 1 n', b=batch_size)
        # Concatenate, 61 (1 cls + 30 mmwave + 30 depth) tokens total
        combined_output = torch.cat((cls, mmWave_output, depth_output), dim=1)
        #combined_output = torch.cat((cls, depth_output), dim=1)
        cls_out = self.combine_features(combined_output)[:, 0] # Get only CLS and use for classification
        return self.output_head(cls_out)
    

class SupervisedUnimodal(nn.Module):
    def __init__(self):
        super(SupervisedUnimodal, self).__init__()
        self.depth_encoder = DepthEncoder()
        self.mmWave_encoder = mmWaveEncoder()
        self.skeleton_encoder = SkeletonEncoder()

        self.mmWave_adapter = nn.Sequential(
                nn.Linear(1920, 800), # used to be 240 maybe that was why it does better?
                nn.ReLU(),
                nn.Linear(800, 27)
            )
        self.depth_adapter = nn.Sequential(
            nn.Linear(1920, 27)
        )
        self.skeleton_adapter = nn.Sequential(
            nn.Linear(480, 240),
            nn.ReLU(),
            nn.Linear(240, 27)
        )
    def forward(self, data, curr_mod):
        if (curr_mod == 'depth'):
            embedding = self.depth_encoder(data)
            batch_size = embedding.shape[0]
            embedding = torch.reshape(embedding, (batch_size, -1))
            return self.depth_adapter(embedding)
        if (curr_mod == 'mmwave'):
            embedding = self.mmwave_encoder(data)
            batch_size = embedding.shape[0]
            embedding = torch.reshape(embedding, (batch_size, -1))
            return self.mmwave_adapter(embedding)
        if (curr_mod == 'skeleton'):
            embedding = self.skeleton_encoder(data)
            batch_size = embedding.shape[0]
            embedding = torch.reshape(embedding, (batch_size, -1))
            return self.skeleton_adapter(embedding)


class SkeletonToDepth(nn.Module):
    def __init__(self):
        super(SkeletonToDepth, self).__init__()
        self.skeleton_encoder = SkeletonEncoder()
        self.depth_decoder = DepthReconstruct()
        self.project = nn.Linear(480, 1920)
    def forward(self, data):
        skeleton_embed = self.skeleton_encoder(data)
        transformed = self.project(skeleton_embed)
        reconstructed_depth = self.depth_decoder(transformed)
        return reconstructed_depth

class SkeletonToMMWave(nn.Module):
    def __init__(self):
        super(SkeletonToMMWave, self).__init__()
        self.skeleton_encoder = SkeletonEncoder()
        self.mmwave_decoder = mmWaveReconstruct()
        self.project = nn.Linear(480, 1920)
    def forward(self, data):
        skeleton_embed = self.skeleton_encoder(data)
        transformed = self.project(skeleton_embed)
        reconstructed_depth = self.mmwave_decoder(transformed)
        return reconstructed_depth
    

# Utilized to train the DualContrastive Learning model
class ContextDualContrastive(nn.Module):
    def __init__(self):
        super(ContextDualContrastive, self).__init__()
        self.mmWave_encoder = mmWaveEncoder()
        self.depth_encoder = DepthEncoder()
        self.skeleton_encoder = SkeletonEncoder()
        self.mmWave_adapter = nn.Sequential(
            nn.Linear(1923, 800), # used to be 240 maybe that was why it does better?
            nn.ReLU(),
            nn.Linear(800, 128)
        )
        self.depth_adapter = nn.Sequential(
            nn.Linear(1923, 800),
            nn.ReLU(),
            nn.Linear(800, 128)
        )
        self.skeleton_adapter = nn.Sequential(
            nn.Linear(483, 240),
            nn.ReLU(),
            nn.Linear(240, 128)
        )
        
    def forward(self, data, mask):
        mmwave_embed = torch.cat((self.mmWave_encoder(data), mask), dim=-1)
        depth_embed = torch.cat((self.depth_encoder(data), mask), dim=-1)
        skeleton_embed = torch.cat((self.skeleton_encoder(data), mask), dim=-1)

        mmWave_output= self.mmWave_adapter(mmwave_embed)
        depth_output = self.depth_adapter(depth_embed)
        skeleton_output = self.skeleton_adapter(skeleton_embed)

        return nn.functional.normalize(depth_output), nn.functional.normalize(mmWave_output),\
            nn.functional.normalize(skeleton_output)