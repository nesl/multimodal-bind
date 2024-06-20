#!/bin/bash

cd ..

for seed in 41 42 43 44 45;
do
    python3 main_baseline3_vector_attach_incomplete_contrastive.py --seed $seed --common_modality acc --gpu 3 --dataset_split split_label0
done

cd ../evaluation
bash eval_baseline3.sh split_label0
cd ../train