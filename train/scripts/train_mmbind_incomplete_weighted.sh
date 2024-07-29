#!/bin/bash
cd ..

for seed in 41 42 43 44 45;
do
    python3 main_mmbind_3_weighted_fuse_contrastive.py --seed $seed --common_modality acc --gpu 3 --dataset_split split_0
    python3 main_mmbind_3_weighted_incomplete_contrastive.py --learning_rate 0.0005 --seed $seed --common_modality acc --gpu 3 --dataset_split split_0
done
