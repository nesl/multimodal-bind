#!/bin/bash

# label ratio from 0.01, 0.05, 0.1, 0.2, 0.5, 1.0
# seed from 101 to 104

for j in 101 102 103 104
do
    for i in 0.01 0.05 0.1 0.2 0.5 1.0
    do
        python3 train.py --gpu 3 --exp_type lowerbound --exp_tag lowerbound --dataset Briareo --batch_size 64 --label_ratio $i --learning_rate 5e-4 --seed $j
    done
done

# python3 train.py --gpu 3 --exp_type lowerbound --exp_tag lowerbound --dataset Briareo --batch_size 32 --label_ratio 0.5 --learning_rate 5e-4 
# python3 train.py --gpu 3 --exp_type lowerbound --exp_tag lowerbound --dataset Briareo --batch_size 32 --label_ratio 0.5 --learning_rate 5e-4
# python3 train.py --gpu 3 --exp_type lowerbound --exp_tag lowerbound --dataset Briareo --batch_size 32 --label_ratio 0.5 --learning_rate 5e-4
# python3 train.py --gpu 3 --exp_type lowerbound --exp_tag lowerbound --dataset Briareo --batch_size 32 --label_ratio 0.5 --learning_rate 5e-4
# python3 train.py --gpu 3 --exp_type lowerbound --exp_tag lowerbound --dataset Briareo --batch_size 32 --label_ratio 0.5 --learning_rate 5e-4