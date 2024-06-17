#!/bin/bash
cd train
python3 main_dualcontrastive_1_train_contrastive.py --seed $1
cd ../evaluation
python3 main_fuse_sup_dual_contrastive.py --seed $1 
cd save_dual_contrastive/results
mv test_accuracy.txt test_accuracy_${1}.txt
