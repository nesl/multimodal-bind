#!/bin/bash

cd ..

python3 main_fuse_sup_baseline1_separate_autoencoder.py --dataset train_C/label_216/

python3 main_fuse_sup_baseline1_separate_autoencoder.py --dataset train_C/label_162/

python3 main_fuse_sup_baseline1_separate_autoencoder.py --dataset train_C/label_108/

python3 main_fuse_sup_baseline1_separate_autoencoder.py --dataset train_C/label_54/