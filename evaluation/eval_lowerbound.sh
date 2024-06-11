# #!/bin/bash

# python3 main_fuse_sup_lower_bound.py --seed $1 --common_modality acc --gpu 3
# python3 main_fuse_sup_lower_bound.py --seed $1 --common_modality gyro --gpu 3
# python3 main_fuse_sup_lower_bound.py --seed $1 --common_modality mag --gpu 3


#!/bin/bash
for seed in 41 42 43 44 45;
do
    # bash ./train_mmbind.sh $seed 0
    python3 main_fuse_sup_lower_bound.py --seed $seed --common_modality acc --gpu 0
    python3 main_fuse_sup_lower_bound.py --seed $seed --common_modality gyro --gpu 0
    python3 main_fuse_sup_lower_bound.py --seed $seed --common_modality mag --gpu 0
done

# ../train/save_mmbind/save_train_AB_contrastive_no_load_mag_41_split_0/models/lr_5e-05_decay_0.0001_bsz_64/last.pt