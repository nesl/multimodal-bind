#!/bin/bash

# seed from 101 to 104

gpu=3
for j in 103 104
do
    for i in 0.05 0.1 0.2 0.5 1.0
    do
        python3 train.py --gpu $gpu --exp_type mmbind --exp_tag label_eval --dataset Briareo --batch_size 64 --learning_rate 5e-4 --weight_decay 5e-3 --seed $j --epochs 250 --label_ratio $i
    done
done