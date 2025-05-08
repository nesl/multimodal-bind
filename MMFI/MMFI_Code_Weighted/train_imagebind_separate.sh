#!/bin/bash
cd train
python3 main_imagebind_1_skeleton_autoencoder.py --seed $1
python3 main_separate_imagebind_2_depth_contrastive.py --seed $1
python3 main_separate_imagebind_2_mmwave_contrastive.py --seed $1
cd ../evaluation
python3 main_fuse_sup_imagebind_separate.py --seed $1 
cd save_imagebind/results
mv test_accuracy.txt test_accuracy_${1}.txt
mv test_f1.txt test_f1_${1}.txt
