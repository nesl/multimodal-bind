#!/bin/bash

cd ..

python3 main_upper_bound_1_single_autoencoder.py --modality gyro
python3 main_upper_bound_1_single_autoencoder.py --modality skeleton

python3 main_upper_bound_2_incomplete_contrastive.py
