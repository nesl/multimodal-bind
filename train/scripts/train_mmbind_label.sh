#!/bin/bash

cd ..

for seed in 41 42 43 44 45;
do
    python3 main_mmbind_label_1_label_pair.py --num_class 7 --gpu 3 --seed $seed --dataset_split split_0 --common_modality acc --learning_rate 1e-5
    python3 main_mmbind_label_1_more_label_pair.py --num_class 7 --gpu 3 --seed $seed --dataset_split split_0 --common_modality acc --learning_rate 1e-5

    python3 main_mmbind_label_2_contrastive_supervise.py --num_class 7 --gpu 3 --seed $seed --dataset_split split_0 --common_modality acc --learning_rate 1e-5
    # python3 main_mmbind_label_2_incomplete_contrastive_supervise.py --num_class 7 --gpu 3 --seed $seed --dataset_split split_0 --common_modality acc --learning_rate 1e-5

    python3 main_mmbind_label_2_contrastive_supervise.py --num_class 7 --gpu 3 --seed $seed --dataset_split split_0 --common_modality acc --learning_rate 1e-5 --use_pair True
    # python3 main_mmbind_label_2_incomplete_contrastive_supervise.py --num_class 7 --gpu 3 --seed $seed --dataset_split split_0 --common_modality acc --learning_rate 1e-5 --use_pair True
done