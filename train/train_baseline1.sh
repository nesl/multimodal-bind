#!/bin/bash

python3 main_baseline1_separate_autoencoder.py --seed $1 --common_modality acc --dataset train_A
python3 main_baseline1_separate_autoencoder.py --seed $1 --common_modality acc --dataset train_B

python3 main_baseline1_separate_autoencoder.py --seed $1 --common_modality gyro --dataset train_A
python3 main_baseline1_separate_autoencoder.py --seed $1 --common_modality gyro --dataset train_B

python3 main_baseline1_separate_autoencoder.py --seed $1 --common_modality mag --dataset train_A
python3 main_baseline1_separate_autoencoder.py --seed $1 --common_modality mag --dataset train_B

cd ../evaluation
bash eval_baseline1.sh $1