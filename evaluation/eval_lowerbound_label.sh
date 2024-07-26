#!/bin/bash
for seed in 41 42 43 44 45;
do
    python3 main_supfuse_lower_bound_label.py --seed $seed --common_modality acc --gpu 3 --dataset_split split_0
done
