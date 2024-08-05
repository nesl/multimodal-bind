#!/bin/bash

cd ..


for seed in 41 42 43 44 45;
do
    # Unimodal encoder training
    python3 main_upper_bound_1_single_autoencoder.py --seed $seed --common_modality gyro --dataset_split split_label0
    python3 main_upper_bound_1_single_autoencoder.py --seed $seed --common_modality mag --dataset_split split_label0

    # Fuse encoders
    python3 main_upper_bound_2_fuse_contrastive_full.py --seed $seed --common_modality acc --gpu 3 --dataset_split split_label0
done