#!/bin/bash

cd ..

for seed in 41 42 43 44 45;
do
    python3 main_baseline2_label_incomplete_multimodal.py --gpu 3 --seed $seed --common_modality acc --dataset_split split_0
done