#!/bin/bash
for seed in 41 42 43 44 45;
do
    python3 main_fuse_sup_upper_bound.py --seed $seed --common_modality acc --learning_rate 1e-5 --weight_decay 1e-3 --gpu 3 --dataset_split $1
done
