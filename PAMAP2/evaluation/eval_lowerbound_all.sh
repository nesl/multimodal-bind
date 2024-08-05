#!/bin/bash

for seed in 41 42 43 44 45;
do
    python3 main_sup_allmod_abc.py --seed $seed --common_modality acc --gpu 3 --dataset_split $1
    python3 main_sup_allmod.py --seed $seed --common_modality acc --gpu 3 --dataset_split $1
done