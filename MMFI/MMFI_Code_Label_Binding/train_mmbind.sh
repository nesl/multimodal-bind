#!/bin/bash
cd train
python3 label_mmbind_1_generate_dataset.py
python3 label_mmbind_2_fuse_contrastive.py --seed $1 
python3 label_mmbind_3_supervised.py --seed $1 

cd ../evaluation
python3 label_mmbind_evaluate.py --seed $1 
rm -r ../train/save_mmbind/train_all_paired_AB
mv save_mmbind/results save_mmbind/results_${1}

