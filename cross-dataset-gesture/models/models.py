import torch
import torch.nn as nn
from models.mod_encoders import SkeletonEncoder, StereoEncoder, DepthEncoder
from models.mod_decoders import SkeletonDecoder, StereoDecoder, DepthDecoder

import utils.log as log

mod_encoder_registry = {"skeleton": SkeletonEncoder, "stereo_ir": StereoEncoder, "depth": DepthEncoder}
mod_decoder_registry = {"skeleton": SkeletonDecoder, "stereo_ir": StereoDecoder, "depth": DepthDecoder}

dims =  {
    "skeleton": 5184,
    "stereo_ir": 1568,
    "depth": 1568
}


def init_model(opt):
    if opt.stage == "eval":
        return GestureMultimodalEncoders(opt)
    else:
        if opt.exp_type == "mmbind":
            if opt.exp_tag == "unimod":
                return UnimodalAutoencoders(opt)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

class UnimodalAutoencoders(nn.Module):
    def __init__(self, opt):
        super(UnimodalAutoencoders, self).__init__()
        self.mod = opt.modality
        self.encoder = mod_encoder_registry[self.mod]()
        self.decoder = mod_decoder_registry[self.mod]()

    def forward(self, batched_data):
        mod_embedding = self.encoder(batched_data[self.mod])
        reconstructed = self.decoder(mod_embedding)
        return reconstructed


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
        
        total_dim = sum([dims[mod]for mod in modalities])
        
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 1280),
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
    
