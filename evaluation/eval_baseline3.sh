#!/bin/bash


for seed in 41 42 43 44 45;
do
    python3 main_fuse_sup_baseline3_vector_attach_incomplete_contrastive.py --common_modality acc --seed $seed --gpu 3 --learning_rate 1e-4 --weight_decay 1e-4 --dataset_split $1 
done
