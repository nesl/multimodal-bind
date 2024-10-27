#!/bin/bash

# seed from 101 to 104

gpu=0
for j in 100 101 102 103 104
do

    # Step 1: label bind different pairings using the pretrained skeleton encoder
    python3 train.py --gpu $gpu --exp_type mmbind --exp_tag label_pair --dataset DHG --batch_size 64 --learning_rate 5e-4 --weight_decay 1e-3 --seed $j --modality skeleton

    # Step 2: pretrain the three modalities encoders on the combined dataset
    python3 train.py --gpu $gpu --exp_type mmbind --exp_tag label_contrastive --dataset DHG --batch_size 64 --learning_rate 5e-4 --weight_decay 1e-3 --seed $j --modality skeleton
done