#!/bin/bash

# seed from 101 to 104

gpu=0
for j in 100 101 102 103 104
do
    # Step 1: unimodal pretrain skeleton encoder
    python3 train.py --gpu $gpu --exp_type mmbind --exp_tag unimod --dataset GR4DHCI --batch_size 64 --learning_rate 5e-4 --weight_decay 1e-3 --seed $j --modality skeleton --epochs 200

    # Step 2: bind different pairings using the pretrained skeleton encoder
    python3 train.py --gpu $gpu --exp_type mmbind --exp_tag pair --dataset DHG --batch_size 64 --learning_rate 5e-4 --weight_decay 1e-3 --seed $j --modality skeleton
    python3 train.py --gpu $gpu --exp_type mmbind --exp_tag pair --dataset GR4DHCI --batch_size 64 --learning_rate 5e-4 --weight_decay 1e-3 --seed $j --modality skeleton

    # Step 3: pretrain the three modalities encoders on the combined dataset
     python3 train.py --gpu 0 --exp_type mmbind --exp_tag contrastive --dataset GR4DHCI --batch_size 64 --learning_rate 5e-4 --weight_decay 1e-3 --seed $j --modality skeleton --epochs 100
done