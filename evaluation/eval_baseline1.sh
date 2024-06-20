#!/bin/bash

for seed in 41 42 43 44 45;
do
    python3 main_fuse_sup_baseline1.py --seed $seed --common_modality acc --dataset_split $1 --gpu 3
done
