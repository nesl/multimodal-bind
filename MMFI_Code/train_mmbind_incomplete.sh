#!/bin/bash
cd train
python3 main_mmbind_1_skeleton_autencoder.py --seed $1
python3 main_mmbind_2_measure_similarity.py --seed $1 
python3 main_mmbind_2_measure_similarity.py --seed $1 --reference_modality mmwave 
cd save_mmbind
mkdir train_all_paired_AB
cp -r train_depth_paired_AB/* train_all_paired_AB
cp -r train_mmwave_paired_AB/* train_all_paired_AB
cd ..


python3 main_mmbind_3_fuse_incomplete_contrastive.py --seed $1

cd ../evaluation
python3 main_fuse_sup_mmbind_incomplete.py --seed $1
cd save_mmbind_incomplete/results
mv test_accuracy.txt test_accuracy_${1}.txt
rm -r /home/jason/Documents/MMBind_MMFI/MMFI_Code/train/save_mmbind/train*

