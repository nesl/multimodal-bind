#!/bin/bash

python3 main_upper_bound_1_single_autoencoder.py --seed $1 --common_modality acc
python3 main_upper_bound_1_single_autoencoder.py --seed $1 --common_modality gyro
python3 main_upper_bound_1_single_autoencoder.py --seed $1 --common_modality mag

python3 main_upper_bound_2_fuse_contrastive.py --seed $1 --common_modality acc
python3 main_upper_bound_2_fuse_contrastive.py --seed $1 --common_modality gyro
python3 main_upper_bound_2_fuse_contrastive.py --seed $1 --common_modality mag

cd ../evaluation
bash eval_upperbound.sh $1