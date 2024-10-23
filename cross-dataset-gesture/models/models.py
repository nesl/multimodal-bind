import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mod_encoders import SkeletonEncoder, StereoEncoder, DepthEncoder
from models.mod_decoders import SkeletonDecoder, StereoDecoder, DepthDecoder

import utils.log as log

mod_encoder_registry = {"skeleton": SkeletonEncoder, "stereo_ir": StereoEncoder, "depth": DepthEncoder}
mod_decoder_registry = {"skeleton": SkeletonDecoder, "stereo_ir": StereoDecoder, "depth": DepthDecoder}

dims = {"skeleton": 5184, "stereo_ir": 1568, "depth": 1568}


def init_model(opt):
    if opt.stage == "eval":
        model = GestureMultimodalEncoders(opt)
        if opt.exp_type == "mmbind":
            # load model weights
            model_weights_path = f"./weights/mmbind/contrastive_{opt.load_pretrain}_{opt.modality}_{opt.seed}_{opt.dataset_split}_1.0/models/lr_0.0005_decay_0.001_bsz_64/last.pth"
            assert os.path.exists(model_weights_path), f"Model weights not found at {model_weights_path}"
            log.logprint(f"Loading model weights from {model_weights_path}")
            pretrained_model_weights = torch.load(model_weights_path)['model']
            model_weights = {}
            for k, v in pretrained_model_weights.items():
                if "encoders" in k and k in model.state_dict():
                    model_weights[k] = v

            model_dict = model.state_dict()
            model_dict.update(model_weights)
            model.load_state_dict(model_dict)
    else:
        if opt.exp_type == "mmbind":
            if opt.exp_tag == "unimod":
                model = UnimodalAutoencoders(opt)
            elif opt.exp_tag == "pair":
                model = UnimodalAutoencoders(opt)
                # load model weights
                model_weights_path = f"./weights/{opt.exp_type}/unimod_{opt.load_pretrain}_{opt.modality}_{opt.seed}_{opt.dataset_split}_{opt.label_ratio}/models/lr_0.0005_decay_0.001_bsz_{opt.batch_size}/best.pth"
                log.logprint(f"Loading model weights from {model_weights_path}")
                assert os.path.exists(model_weights_path), f"Model weights not found at {model_weights_path}"

                model_weights = torch.load(model_weights_path)
                model.load_state_dict(model_weights["model"])
            elif opt.exp_tag == "contrastive":
                model = GestureMultimodalEncoders(opt)
        else:
            raise NotImplementedError
    model = model.cuda()
    return model


class UnimodalAutoencoders(nn.Module):
    def __init__(self, opt):
        super(UnimodalAutoencoders, self).__init__()
        self.mod = opt.modality
        self.encoder = mod_encoder_registry[self.mod]()
        self.decoder = mod_decoder_registry[self.mod]()

    def forward(self, batched_data):
        mod_embedding = self.encoder(batched_data[self.mod])
        reconstructed = self.decoder(mod_embedding)
        return mod_embedding, reconstructed


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

        total_dim = sum([dims[mod] for mod in modalities])

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_class),
        )

        if opt.stage == "eval":
            self.decode = True
        else:
            self.decode = False
            self.mod_projector_head = nn.ModuleDict()
            for mod in modalities:
                self.mod_projector_head[mod] = nn.Sequential(
                    nn.Linear(dims[mod], 1280),
                    nn.BatchNorm1d(1280),
                    nn.ReLU(inplace=True),
                    nn.Linear(1280, 128),
                )

    def forward(self, batched_data):
        mod_embeddings = []
        for mod in batched_data:
            if mod not in self.encoders:
                continue
            mod_embedding = self.encoders[mod](batched_data[mod])
            mod_embeddings.append(mod_embedding)

        if self.decode:
            mod_embeddings = torch.cat(mod_embeddings, dim=1)
            logits = self.classifier(mod_embeddings)
            return logits

        projected_embeddings = []
        for i, mod in enumerate(batched_data):
            if mod not in self.encoders:
                continue
            projected_embeddings.append(F.normalize(self.mod_projector_head[mod](mod_embeddings[i]), dim=1))

        return projected_embeddings
