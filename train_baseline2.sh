#!/bin/bash
cd train
python3 main_baseline2_mask_contrastive.py --seed $1
cd ../evaluation
python3 main_fuse_sup_baseline2.py --seed $1
cd save_train_C/label_216/baseline2/results
mv test_accuracy.txt test_accuracy_${1}.txt
