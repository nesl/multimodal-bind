import torch

def calc_loss(opt, model, data, labels, loss_func):
    if opt.exp_type == "mmbind":
        if opt.exp_tag == "unimod":
            mod = opt.modality
            mod_embedding, pred = model(data)
            gt = data[mod]
            return mod_embedding, loss_func(pred, gt)
        elif opt.exp_tag == "pair":
            mod_embedding, pred = model(data)
            return mod_embedding, torch.tensor(-1)
        elif opt.exp_tag in {"contrastive", "label_contrastive"}:
            mod_embedding = model(data)
            mod_embedding = torch.stack(mod_embedding, dim=1)
            loss = loss_func(mod_embedding)
            return mod_embedding, loss
            
    else:
        raise NotImplementedError