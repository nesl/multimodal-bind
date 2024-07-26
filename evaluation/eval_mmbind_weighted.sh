#!/bin/bash

for seed in 43;
do
    python3 main_fuse_sup_mmbind_weighted.py --seed $seed --common_modality acc --learning_rate 5e-5 --weight_decay 1e-4 --gpu 3 --dataset_split split_0

    # python3 main_fuse_sup_mmbind_weighted.py --seed $seed --common_modality acc --learning_rate 1e-4 --weight_decay 1e-4 --gpu 3 --dataset_split split_0 --load_pretrain load_pretrain
done
