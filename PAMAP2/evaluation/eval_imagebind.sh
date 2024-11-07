#!/bin/bash


for seed in 41 42 43 44 45;
do
    python3 main_fuse_sup_imagebind.py --seed $seed --common_modality acc --learning_rate 1e-4 --weight_decay 1e-4
done
