from torch.utils.data import DataLoader

from param.parse_args import parse_option
from models.models import GestureMultimodalEncoders
from models.loss import init_loss
from data.gesture_dataset import MultimodalDataset

from train_utils.train_engine import train_engine
from evaluation.eval_engine import eval_engine

def train(opt):
    model = GestureMultimodalEncoders(opt)
    loss_func = init_loss(opt)
    
    model = model.cuda()
    loss_func = loss_func.cuda()
    
    train_dataset = MultimodalDataset(opt)
    val_dataset = MultimodalDataset(opt, mode="valid")
    
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
    
    if opt.stage == "train":
        train_engine(opt, model, loss_func, train_dataloader, val_dataloader)
    else:
        eval_engine(opt, model, loss_func, train_dataloader, val_dataloader)

if __name__ == '__main__':
    opt = parse_option()
    train(opt)
    