import torch
import torch.nn as nn
from models.mod_encoders import SkeletonEncoder, StereoEncoder, DepthEncoder
from models.mod_decoders import SkeletonDecoder, StereoDecoder, DepthDecoder

import utils.log as log

mod_encoder_registry = {"skeleton": SkeletonEncoder, "stereo_ir": StereoEncoder, "depth": DepthEncoder}
mod_decoder_registry = {"skeleton": SkeletonDecoder, "stereo_ir": StereoDecoder, "depth": DepthDecoder}

# mod_encoder_registry = {"skeleton": SkeletonEncoder}
# mod_encoder_registry = {"stereo_ir": StereoEncoder}
# mod_encoder_registry = {"depth": DepthEncoder}


class UnimodalAutoencoders(nn.Module):
    def __init__(self, mod):
        super(UnimodalAutoencoders, self).__init__()
        mod = self.get_mod()
        self.encoder = mod_encoder_registry[mod]()
        self.decoder = mod_decoder_registry[mod]()
        raise NotImplementedError

    def forward(self, batched_data):
        raise NotImplementedError


class GestureMultimodalEncoders(nn.Module):
    def __init__(self, opt):
        super(GestureMultimodalEncoders, self).__init__()
        log.logprint()
        log.logprint("Initializing Gesture Multimodal Encoders")
        modalities = mod_encoder_registry.keys()
        log.logprint(f"Modalities: {modalities}")
        log.logprint(f"Number of classes: {opt.num_class}")
        num_class = opt.num_class

        self.encoders = nn.ModuleDict({mod: mod_encoder_registry[mod]() for mod in modalities})

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_class),
        )
        
        self.decode = True

    def forward(self, batched_data):
        mod_embeddings = []
        for mod in batched_data:
            if mod not in self.encoders:
                continue
            mod_embedding = self.encoders[mod](batched_data[mod])
            mod_embeddings.append(mod_embedding)

        mod_embeddings = torch.cat(mod_embeddings, dim=1)
        
        
        if self.decode:
            logits = self.classifier(mod_embeddings)
            return logits
        
        return mod_embeddings
    
