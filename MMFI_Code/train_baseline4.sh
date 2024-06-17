#!/bin/bash
cd train
python3 main_baseline4_1_train_gen_model.py --seed $1
python3 main_baseline4_1_train_gen_model.py --seed $1 --dataset train_B 
python3 main_baseline4_2_generate_missing.py --seed $1
python3 main_baseline4_3_fuse_contrastive.py --seed $1  
cd ../evaluation
python3 main_fuse_sup_baseline4.py --seed $1 
cd save_baseline4/results
mv test_accuracy.txt test_accuracy_${1}.txt
rm -r /home/jason/Documents/MMBind_MMFI/MMFI_Code/train/save_baseline4/train_paired_AB

