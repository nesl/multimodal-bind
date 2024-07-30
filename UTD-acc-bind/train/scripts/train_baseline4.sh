#!/bin/bash

cd ..

python3 main_baseline4_1_cross_encoder.py --dataset train_A
python3 main_baseline4_1_cross_encoder.py --dataset train_B

python3 main_baseline4_2_cross_generation.py --dataset train_A
python3 main_baseline4_2_cross_generation.py --dataset train_B

python3 main_baseline4_3_fuse_contrastive.py
