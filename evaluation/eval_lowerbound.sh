# #!/bin/bash

# python3 main_fuse_sup_lower_bound.py --seed $1 --common_modality acc --gpu 3
# python3 main_fuse_sup_lower_bound.py --seed $1 --common_modality gyro --gpu 3
# python3 main_fuse_sup_lower_bound.py --seed $1 --common_modality mag --gpu 3


#!/bin/bash
for seed in 41 42 43 44 45;
do
    python3 main_fuse_sup_lower_bound_abc.py --seed $seed --common_modality acc --gpu 3 --dataset_split split_0 --learning_rate 0.00005
    python3 main_fuse_sup_lower_bound.py --seed $seed --common_modality acc --gpu 3 --dataset_split split_0 --learning_rate 0.00005
    # python3 main_fuse_sup_lower_bound_abc.py --seed $seed --common_modality gyro --gpu 3 --dataset_split split_0
    # python3 main_fuse_sup_lower_bound_abc.py --seed $seed --common_modality mag --gpu 3 --dataset_split split_0
done

# for seed in 41 42 43 44 45;
# do
#     python3 main_sup_unimod.py --seed $seed --common_modality acc --learning_rate 1e-4 --weight_decay 1e-4 --gpu 3 --dataset_split split_label0
#     python3 main_sup_unimod.py --seed $seed --common_modality gyro --learning_rate 1e-4 --weight_decay 1e-4 --gpu 3 --dataset_split split_label0
#     python3 main_sup_unimod.py --seed $seed --common_modality mag --learning_rate 1e-4 --weight_decay 1e-4 --gpu 3 --dataset_split split_label0
# done
