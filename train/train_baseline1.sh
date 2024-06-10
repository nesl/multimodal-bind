#!/bin/bash

python3 main_baseline1_separate_autoencoder.py --common_modality acc --dataset train_A
python3 main_baseline1_separate_autoencoder.py --common_modality acc --dataset train_B

python3 main_baseline1_separate_autoencoder.py --common_modality gyro --dataset train_A
python3 main_baseline1_separate_autoencoder.py --common_modality gyro --dataset train_B

python3 main_baseline1_separate_autoencoder.py --common_modality mag --dataset train_A
python3 main_baseline1_separate_autoencoder.py --common_modality mag --dataset train_B