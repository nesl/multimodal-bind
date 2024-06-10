#!/bin/bash
cd train
python3 main_upper_bound_1_single_autoencoder.py --seed $1
python3 main_upper_bound_1_single_autoencoder.py --seed $1 --modality mag
python3 main_upper_bound_2_fuse_contrastive.py --seed $1 

cd ../evaluation
python3 main_fuse_sup_upper_bound.py --seed $1
cd save_train_C/label_216/upper_bound/results
mv test_accuracy.txt test_accuracy_${1}.txt
