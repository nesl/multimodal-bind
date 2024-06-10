#!/bin/bash

# Step 1: train unimodal encoders
python3 main_mmbind_1_unimod_autoencoder.py --seed 123 --common_modality acc
python3 main_mmbind_1_unimod_autoencoder.py --seed 123 --common_modality acc
python3 main_mmbind_1_unimod_autoencoder.py --seed 123 --common_modality acc


# Step 2: measure similarity

# remove duplicate files
cd save_mmbind
rm -rf train_setA_paired_AB_acc
rm -rf train_setB_paired_AB_acc
rm -rf train_setA_paired_AB_gyro
rm -rf train_setB_paired_AB_gyro
rm -rf train_setA_paired_AB_mag
rm -rf train_setB_paired_AB_mag

rm -rf train_all_paired_AB_acc
rm -rf train_all_paired_AB_gyro
rm -rf train_all_paired_AB_mag

# similarity measurement
python3 main_mmbind_2_measure_similarity.py --seed 123 --common_modality acc --reference_modality setA
python3 main_mmbind_2_measure_similarity.py --seed 123 --common_modality acc --reference_modality setB

python3 main_mmbind_2_measure_similarity.py --seed 123 --common_modality gyro --reference_modality setA
python3 main_mmbind_2_measure_similarity.py --seed 123 --common_modality gyro --reference_modality setB


python3 main_mmbind_2_measure_similarity.py --seed 123 --common_modality mag --reference_modality setA
python3 main_mmbind_2_measure_similarity.py --seed 123 --common_modality mag --reference_modality setB

# mv files
cd save_mmbind
mkdir train_all_paired_AB_acc
cp train_setA_paired_AB_acc/* train_all_paired_AB_acc
cp train_setB_paired_AB_acc/* train_all_paired_AB_acc

mkdir train_all_paired_AB_gyro
cp train_setA_paired_AB_gyro/* train_all_paired_AB_gyro
cp train_setB_paired_AB_gyro/* train_all_paired_AB_gyro

mkdir train_all_paired_AB_mag
cp train_setA_paired_AB_mag/* train_all_paired_AB_mag
cp train_setB_paired_AB_mag/* train_all_paired_AB_mag
cd ..

# run contrastive
python3 main_mmbind_3_fuse_contrastive.py --seed 123 --common_modality acc
python3 main_mmbind_3_fuse_contrastive.py --seed 123 --common_modality gyro
python3 main_mmbind_3_fuse_contrastive.py --seed 123 --common_modality mag