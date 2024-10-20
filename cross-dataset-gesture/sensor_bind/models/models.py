import torch.nn as nn
from models.mod_encoders import SkeletonEncoder, StereoEncoder, DepthEncoder
from models.mod_decoders import SkeletonDecoder, StereoDecoder, DepthDecoder

mod_encoder_registry = {
    "skeleton": SkeletonEncoder,
    "stereo_ir": StereoEncoder,
    "depth": DepthEncoder
}

mod_decoder_registry = {
    "skeleton": SkeletonDecoder,
    "stereo_ir": StereoDecoder,
    "depth": DepthDecoder
}

class UnimodalAutoencoders(nn.Module):
    def __init__(self):
        super(UnimodalAutoencoders, self).__init__()
        mod = self.get_mod()
        self.encoder = mod_encoder_registry[mod]()
        self.decoder = mod_decoder_registry[mod]()
        raise NotImplementedError

    def forward(self, batched_data):
        raise NotImplementedError
