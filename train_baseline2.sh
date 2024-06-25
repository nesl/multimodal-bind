#!/bin/bash
cd train
python3 main_baseline2_mask_contrastive.py --seed $1 
cd ../evaluation
python3 main_fuse_sup_baseline2.py --seed $1
cd save_baseline2/results
mv test_accuracy.txt test_accuracy_${1}.txt
mv test_f1.txt test_f1_${1}.txt
