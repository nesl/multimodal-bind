#!/bin/bash

cd ..

for seed in 41 42 43 44 45;
do
    python3 main_baseline1_label_unimodal_supervise.py --gpu 3 --seed $seed --common_modality gyro --dataset_split split_0
    python3 main_baseline1_label_unimodal_supervise.py --gpu 3 --seed $seed --common_modality mag --dataset_split split_0
done