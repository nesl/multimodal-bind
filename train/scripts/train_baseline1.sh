#!/bin/bash

cd ..

for seed in 41 42 43 44 45;
do
    # python3 main_baseline1_separate_autoencoder.py --seed $seed --common_modality acc --dataset train_A --gpu 3 --dataset_split split_label0
    # python3 main_baseline1_separate_autoencoder.py --seed $seed --common_modality acc --dataset train_B --gpu 3 --dataset_split split_label0
    python3 main_baseline1_separate_autoencoder.py --seed $seed --common_modality gyro --dataset train_A --gpu 3 --dataset_split split_label0
    python3 main_baseline1_separate_autoencoder.py --seed $seed --common_modality gyro --dataset train_B --gpu 3 --dataset_split split_label0
    python3 main_baseline1_separate_autoencoder.py --seed $seed --common_modality mag --dataset train_A --gpu 3 --dataset_split split_label0
    python3 main_baseline1_separate_autoencoder.py --seed $seed --common_modality mag --dataset train_B --gpu 3 --dataset_split split_label0
done

cd ../evaluation
bash eval_baseline1.sh split_label0
cd ../train/scripts