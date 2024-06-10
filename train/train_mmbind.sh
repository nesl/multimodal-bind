#!/bin/bash

# Step 1: train unimodal encoders
python3 main_mmbind_1_unimod_autoencoder.py --seed $1 --common_modality acc
python3 main_mmbind_1_unimod_autoencoder.py --seed $1 --common_modality acc
python3 main_mmbind_1_unimod_autoencoder.py --seed $1 --common_modality acc


# Step 2: measure similarity

# remove duplicate files
cd save_mmbind
rm -rf train_setA_paired_AB_acc_$1_split_0
rm -rf train_setB_paired_AB_acc_$1_split_0
rm -rf train_setA_paired_AB_gyro_$1_split_0
rm -rf train_setB_paired_AB_gyro_$1_split_0
rm -rf train_setA_paired_AB_mag_$1_split_0
rm -rf train_setB_paired_AB_mag_$1_split_0

rm -rf train_all_paired_AB_acc_$1_split_0
rm -rf train_all_paired_AB_gyro_$1_split_0
rm -rf train_all_paired_AB_mag_$1_split_0

# similarity measurement
python3 main_mmbind_2_measure_similarity.py --seed $1 --common_modality acc --reference_modality setA
python3 main_mmbind_2_measure_similarity.py --seed $1 --common_modality acc --reference_modality setB

python3 main_mmbind_2_measure_similarity.py --seed $1 --common_modality gyro --reference_modality setA
python3 main_mmbind_2_measure_similarity.py --seed $1 --common_modality gyro --reference_modality setB


python3 main_mmbind_2_measure_similarity.py --seed $1 --common_modality mag --reference_modality setA
python3 main_mmbind_2_measure_similarity.py --seed $1 --common_modality mag --reference_modality setB

# mv files
cd save_mmbind
mkdir train_all_paired_AB_acc_$1_split_0
cp train_setA_paired_AB_acc_$1_split_0/* train_all_paired_AB_acc_$1_split_0
cp train_setB_paired_AB_acc_$1_split_0/* train_all_paired_AB_acc_$1_split_0

mkdir train_all_paired_AB_gyro_$1_split_0
cp train_setA_paired_AB_gyro_$1_split_0/* train_all_paired_AB_gyro_$1_split_0
cp train_setB_paired_AB_gyro_$1_split_0/* train_all_paired_AB_gyro_$1_split_0

mkdir train_all_paired_AB_mag_$1_split_0
cp train_setA_paired_AB_mag_$1_split_0/* train_all_paired_AB_mag_$1_split_0
cp train_setB_paired_AB_mag_$1_split_0/* train_all_paired_AB_mag_$1_split_0
cd ..

# run contrastive
python3 main_mmbind_3_fuse_contrastive.py --seed $1 --common_modality acc
python3 main_mmbind_3_fuse_contrastive.py --seed $1 --common_modality gyro
python3 main_mmbind_3_fuse_contrastive.py --seed $1 --common_modality mag

cd ../evaluation
bash eval.sh