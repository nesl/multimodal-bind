#!/bin/bash
cd train
python3 main_mmbind_1_wifi_autencoder.py --seed $1
python3 main_mmbind_2_measure_similarity_random.py --seed $1 
python3 main_mmbind_2_measure_similarity_random.py --seed $1 --reference_modality mmwave 
cd save_mmbind
mkdir train_all_paired_AB
cp -r train_depth_paired_AB/* train_all_paired_AB
cp -r train_mmwave_paired_AB/* train_all_paired_AB
cd ..


python3 main_mmbind_3_fuse_contrastive.py --seed $1

cd ../evaluation
python3 main_fuse_sup_mmbind.py --seed $1
cd save_mmbind/results
mv test_accuracy.txt test_accuracy_${1}.txt
rm -r ../../../train/save_mmbind/train*

