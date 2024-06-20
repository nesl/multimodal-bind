import glob
import os
import random
import numpy as np

label_class_dir = os.listdir("../processed_data_all")
c = [int(l.split("_")[1]) for l in label_class_dir]
# c -> freq
label_count = {i: c.count(i) for i in set(c)}
label_sum = sum(label_count.values())
print(label_count)


splits = ["train_A", "train_B", "test", "train_C"]

average = label_sum // len(splits)


label_classes = list(sorted(set([int(sub.split('_')[1]) for sub in os.listdir("../processed_data_all")])))
print(label_classes)

shuffle_classes = random.sample(label_classes, len(label_classes))


indice_dir = f"../indices"
indice_gen_id = len(os.listdir(indice_dir))

indice_id_dir = os.path.join(indice_dir, f"split_label0")
if not os.path.exists(indice_id_dir):
    os.makedirs(indice_id_dir)


idx = 0
split_info = []
for i, split in enumerate(splits):
    split_count = 0
    label_i = []
    while abs(split_count - average) > 50 and split_count < average and len(shuffle_classes) > 0:
        label = shuffle_classes.pop()
        label_i.append(label)
        split_count += label_count[label]
    
    print(f"{split}: {label_i}, {split_count}")
    split_info.append(f"{split}: {label_i} ({split_count})")

    subject_i_indices = []
    for subject in label_i:
        subject_i_indices += glob.glob(f"../processed_data_all/subject*_{subject}_*.npy")
    
    print(split, len(subject_i_indices))
    
    subject_i_indices = [i.split("/")[-1] for i in subject_i_indices]

    # np.savetxt(os.path.join(indice_id_dir, f"{split}.txt"), subject_i_indices, delimiter="\n", fmt="%s")
    np.savetxt(os.path.join(indice_id_dir, f"{split}.txt"), subject_i_indices, delimiter="\n", fmt="%s")

np.savetxt(os.path.join(indice_id_dir, f"split_info.txt"), split_info, delimiter="\n", fmt="%s")

