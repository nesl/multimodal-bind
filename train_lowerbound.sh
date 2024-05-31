#!/bin/bash
cd evaluation
python3 main_fuse_sup_lower_bound.py --seed $1
cd save_train_C/label_216/supfuse_lower_bound/results
mv test_accuracy.txt test_accuracy_${1}.txt
