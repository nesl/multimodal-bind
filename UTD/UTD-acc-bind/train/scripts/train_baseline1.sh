#!/bin/bash

cd ..

python3 main_baseline1_separate_autoencoder.py --dataset train_A
python3 main_baseline1_separate_autoencoder.py --dataset train_B
