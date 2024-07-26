#!/bin/bash

cd ..

for seed in 43 44 45;
do
    python3 main_upper_bound_label.py --gpu 3 --seed $seed --common_modality acc --dataset_split split_0
    python3 main_upper_bound_label_contrastive_supervise.py --gpu 3 --seed $seed --common_modality acc --dataset_split split_0
done