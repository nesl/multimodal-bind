import torch
import models.models
from shared_files.data_pre import TrainA, TrainB
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def save_img(images_tensor):
    for i in range(images_tensor.size(0)):
        # Get each image and convert to CPU if necessary
        img = images_tensor[i].permute(2, 1, 0).cpu()  # Convert from [3, 640, 480] to [640, 480, 3]
        img = (img * 255).byte()  # Scale from [0,1] to [0,255] and convert to 8-bit
        img_pil = Image.fromarray(img.numpy())

        # Save image with a unique name
        img_pil.save(f"./images/image_{i}.png")
ae_model = models.models.ImageAE()
ae_model.load_state_dict(torch.load('./save_mmbind/save_train_AB_acc_AE/models/single_train_AB_lr_0.0001_decay_0.0001_bsz_64/last.pth')['model'])
#ae_model.load_state_dict(torch.load('./save_mmbind/save_train_AB_acc_AE/models/Good32/last.pth')['model'])
img_dataset = torch.utils.data.ConcatDataset([TrainA(), TrainB()])
img_dataloader = DataLoader(img_dataset, batch_size=64, num_workers=20)

embed_arr = []
label_arr = []
ae_model.cuda()
i = 0
with torch.no_grad():
    for batched_data in img_dataloader:
        print(i, len(img_dataloader))
        i += 1
        for key in batched_data:
            batched_data[key] = batched_data[key].cuda()
        bsz = len(batched_data['label'])
        embed = ae_model.enc(batched_data['img'])
        embed_arr.append(torch.reshape(embed, (bsz, -1)))
        label_arr += batched_data['label']
    embed_npy = torch.concat(embed_arr, dim=0).cpu().detach().numpy()

print("Calculating TSNE")
tsne_out = TSNE().fit_transform(embed_npy)
sns.set_theme(font_scale=2)
sns.scatterplot(x=tsne_out[:, 0], y=tsne_out[:, 1], hue=label_arr, s=200, palette='colorblind')
plt.show()


