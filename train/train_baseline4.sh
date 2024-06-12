#!/bin/bash

for seed in 41 42 43 44 45;
do
    python3 main_baseline4_1_cross_encoder.py --seed $seed --common_modality acc --gpu 3 --dataset train_A
    python3 main_baseline4_1_cross_encoder.py --seed $seed --common_modality acc --gpu 3 --dataset train_B
done
