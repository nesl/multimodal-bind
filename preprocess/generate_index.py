import glob
import os
import random
import numpy as np


processed_dir = "../processed_data"
valid_actions = [1, 3, 4, 12, 13, 16, 17]

subjects = [sub.split('_')[0] for sub in os.listdir(processed_dir)]
subjects = list(set([sub for sub in subjects if "subject" in sub])) # assert all sub and remove duplicates

shuffle_subjects = random.sample(subjects, len(subjects))

splits = ["train_A", "train_B", "train_C", "test"]

num_sub_per_split = len(subjects) // len(splits)


indice_dir = f"./indices"
indice_gen_id = len(os.listdir(indice_dir))

indice_id_dir = os.path.join(indice_dir, f"split_{indice_gen_id}")
if not os.path.exists(indice_id_dir):
    os.makedirs(indice_id_dir)

split_info = []
for i, split in enumerate(splits):
    subject_i = shuffle_subjects[i*num_sub_per_split:(i+1)*num_sub_per_split if i != len(splits) - 1 else None] # test has all rest
    split_info.append(f"{split}: {subject_i}")

    subject_i_indices = []
    for subject in subject_i:
        subject_i_indices += glob.glob(f"./{processed_dir}/{subject}_*.npy")
    
    subject_i_indices = [i.split("/")[-1] for i in subject_i_indices]

    np.savetxt(os.path.join(indice_id_dir, f"{split}.txt"), subject_i_indices, delimiter="\n", fmt="%s")


np.savetxt(os.path.join(indice_id_dir, f"split_info.txt"), split_info, delimiter="\n", fmt="%s")


