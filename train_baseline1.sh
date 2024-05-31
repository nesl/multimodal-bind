#!/bin/bash
cd train
python3 main_baseline1_separate_autoencoder.py --seed $1
python3 main_baseline1_separate_autoencoder.py --dataset train_B --seed $1
cd ../evaluation
python3 main_fuse_sup_baseline1.py --seed $1
cd save_train_C/label_216/baseline1/results
mv test_accuracy.txt test_accuracy_${1}.txt
