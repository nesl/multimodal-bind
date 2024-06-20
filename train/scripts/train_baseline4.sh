#!/bin/bash

cd ..

for seed in 41 42 43 44 45;
do
    # python3 main_baseline4_1_cross_encoder.py --seed $seed --common_modality acc --gpu 3 --dataset train_A --dataset_split split_label0
    # python3 main_baseline4_1_cross_encoder.py --seed $seed --common_modality acc --gpu 3 --dataset train_B --dataset_split split_label0

    # python3 main_baseline4_2_cross_generation.py --seed $seed --common_modality acc --gpu 3 --dataset train_A --dataset_split split_label0
    # python3 main_baseline4_2_cross_generation.py --seed $seed --common_modality acc --gpu 3 --dataset train_B --dataset_split split_label0

    python3 main_baseline4_3_fuse_contrastive.py --seed $seed --common_modality acc --gpu 3  --dataset_split split_label0
done

cd ../evaluation
bash eval_baseline4.sh split_label0
cd ../train