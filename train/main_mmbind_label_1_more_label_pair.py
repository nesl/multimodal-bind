import os

import numpy as np
import torch

from shared_files import data_pre as data


from modules.option_utils import parse_option
from modules.print_utils import pprint

def load_single_modal_set(opt, mod, root):

    print(f"=\tLoading data {mod} from {root}")
    train_dataset = data.Multimodal_dataset([], [mod], root=root, opt=opt)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    
    x = []
    y = []
    for _, batch in enumerate(train_loader):
        batch_data = batch[mod]
        labels = batch['action']

        x.append(batch_data)
        y.append(labels)
    
    # [nb] -> [nb * b]
    x = torch.concatenate(x, dim=0)
    y = torch.concatenate(y, dim=0)

    return x, y


def label_pair_data(opt):

    #load labeled train and test data
    x1_A, y_A = load_single_modal_set(opt, opt.mod1, "train_A")
    x2_B, y_B = load_single_modal_set(opt, opt.mod2, "train_B")

    count_class = np.zeros((opt.num_class, 2))
    index_A = []
    index_B = []

    for class_id in range(opt.num_class):
        count_class[class_id, 0] = np.count_nonzero(y_A == class_id)
        count_class[class_id, 1] = np.count_nonzero(y_B == class_id)

        index_A.append(np.where(y_A == class_id)[0])
        index_B.append(np.where(y_B == class_id)[0])

    print(f"=\t{y_A.shape}, {y_B.shape}, {count_class}")
    pprint(f"{y_A.shape}, {y_B.shape}, {count_class}")

    x1 = []
    x2 = []
    y = []

    for class_id in range(opt.num_class):#opt.num_class

        for sample_id in range(index_A[class_id].shape[0]):

            sample_index = index_A[class_id][sample_id]
            # print(y_A[sample_index])

            for random_id in range(index_B[class_id].shape[0]):
                random_index = index_B[class_id][random_id]
                x1.append(x1_A[sample_index])## skeleton
                y.append(y_A[sample_index])
                x2.append(x2_B[random_index])#acc
            
        for sample_id in range(index_B[class_id].shape[0]):

            sample_index = index_B[class_id][sample_id]
            # print(y_B[sample_index])

            for random_id in range(index_A[class_id].shape[0]):
                random_index = index_A[class_id][random_id]
                x2.append(x2_B[sample_index])## gyro
                y.append(y_B[sample_index])
                x1.append(x1_A[random_index])#skeleton

    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    pprint(y)
    pprint(f"{x1.shape}, {x2.shape}, {y.shape}")


    folder_path = f"./save_mmbind_more_label_paired_data_{opt.common_modality}_{opt.seed}_{opt.dataset_split}/"

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    np.save(folder_path + f'{opt.mod1}.npy', x1)
    np.save(folder_path + f'{opt.mod2}.npy', x2)
    np.save(folder_path + 'label.npy', y)


def main():
    opt = parse_option("save_mmbind_label_bind", "more_label_pair")
    
    common_modality = opt.common_modality
    other_modalities = [m for m in ['acc', 'gyro', 'mag'] if m != common_modality]
    opt.mod1 = other_modalities[0]
    opt.mod2 = other_modalities[1]

    pprint(f"Common modality: {opt.common_modality}")
    pprint(f"Mod1: {opt.mod1}")
    pprint(f"Mod2: {opt.mod2}")

    print(f"=\tCommon modality: {opt.common_modality}")
    print(f"=\tMod1: {opt.mod1}")
    print(f"=\tMod2: {opt.mod2}")

    label_pair_data(opt)


if __name__ == '__main__':
    main()
