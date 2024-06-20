#!/bin/bash

cd ..

for seed in 41 42 43 44 45;
do
    python3 main_dualcontrastive_1_train_contrastive.py --seed $seed --common_modality acc --gpu 3 --dataset_split split_label0
done

cd ../evaluation
bash eval_dual.sh split_label0
cd ../train