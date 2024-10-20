from torch.utils.data import DataLoader

from param.parse_args import parse_args
from models.models import MMBind
from data.gesture_dataset import MultimodalDataset

from train.train_engine import train_engine
from evaluation.eval_engine import eval_engine

def train(opt):
    model = MMBind()
    dataset = MultimodalDataset(valid_actions=opt.valid_actions, valid_mods=opt.valid_mods)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    
    if opt.stage == "train":
        train_engine(opt, model, dataloader)
    else:
        eval_engine(opt, model, dataloader)

if __name__ == '__main__':
    opt = parse_args()
    train(opt)
    