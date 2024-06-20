#!/bin/bash

cd ..


# for seed in 41 42 43 44 45;
# do
#     python3 main_upper_bound_1_single_autoencoder.py --seed $seed --common_modality gyro --dataset_split split_label0
#     python3 main_upper_bound_1_single_autoencoder.py --seed $seed --common_modality mag --dataset_split split_label0

#     python3 main_upper_bound_2_fuse_contrastive.py --seed $seed --common_modality acc --gpu 3 --learning_rate 0.00001 --dataset_split split_label0
# done

cd ../evaluation
bash eval_upperbound.sh split_label0
cd ../train