from param.parse_args import parse_option
from models.models import init_model
from models.loss import init_loss
from data.gesture_dataset import init_dataloader

from train_utils.train_engine import train_engine
from train_utils.pair_engine import pair_engine
from train_utils.eval_engine import eval_engine

def train(opt):
    model = init_model(opt)
    loss_func = init_loss(opt)
    
    train_dataloader = init_dataloader(opt, mode="train")
    val_dataloader = init_dataloader(opt, mode="valid")

    if opt.stage == "train":
        if opt.exp_tag == "pair":
            pair_engine(opt, model, loss_func, train_dataloader)
        else:
            train_engine(opt, model, loss_func, train_dataloader, val_dataloader)
    else:
        eval_engine(opt, model, loss_func, train_dataloader, val_dataloader)

if __name__ == '__main__':
    opt = parse_option()
    train(opt)
    