#!/bin/bash
cd evaluation
python3 main_fuse_sup_lower_bound.py --seed $1
cd save_lower_bound/results
mv test_accuracy.txt test_accuracy_${1}.txt
mv test_f1.txt test_f1_${1}.txt