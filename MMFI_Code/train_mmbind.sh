#!/bin/bash
cd train
python3 main_mmbind_1_acc_autencoder.py --seed $1
python3 main_mmbind_2_measure_similarity.py --seed $1
python3 main_mmbind_2_measure_similarity.py --seed $1 --reference_modality mag
cd save_mmbind
mkdir train_all_paired_AB
cp train_gyro_paired_AB/* train_all_paired_AB
cp train_mag_paired_AB/* train_all_paired_AB
cd ..


python3 main_mmbind_3_fuse_contrastive.py --seed $1

cd ../evaluation
python3 main_fuse_sup_mmbind.py --seed $1
cd save_mmbind/results
mv test_accuracy.txt test_accuracy_${1}.txt
cd /home/jason/Documents/MMBind_MMFI/MMFI_Code/train/save_mmbind
rm -r train*
