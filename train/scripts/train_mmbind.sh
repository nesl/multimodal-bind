#!/bin/bash
cd ..

for seed in 41 42 43 44 45;
do
    # python3 main_mmbind_1_unimod_autencoder.py --seed $seed --common_modality acc --gpu 3 --dataset_split split_label0

    # remove duplicate files
    # cd save_mmbind
    # rm -rf "train_setA_paired_AB_acc_"$seed"_split_label0"
    # rm -rf "train_setB_paired_AB_acc_"$seed"_split_label0"
    # rm -rf "train_setA_paired_AB_gyro_"$seed"_split_label0"
    # rm -rf "train_setB_paired_AB_gyro_"$seed"_split_label0"
    # rm -rf "train_setA_paired_AB_mag_"$seed"_split_label0"
    # rm -rf "train_setB_paired_AB_mag_"$seed"_split_label0"

    # rm -rf "train_all_paired_AB_acc_"$seed"_split_label0"
    # rm -rf "train_all_paired_AB_gyro_"$seed"_split_label0"
    # rm -rf "train_all_paired_AB_mag_"$seed"_split_label0"
    # cd ..

    # echo "Begin measure similarity"

    # python3 main_mmbind_2_measure_similarity.py --seed $seed --common_modality acc --reference_modality setA --dataset_split split_label0
    # python3 main_mmbind_2_measure_similarity.py --seed $seed --common_modality acc --reference_modality setB --dataset_split split_label0

    # # cd save_mmbind
    # cd save_mmbind
    # mkdir "train_all_paired_AB_acc_"$seed"_split_label0"
    # cp "train_setA_paired_AB_acc_"$seed"_split_label0"/* "train_all_paired_AB_acc_"$seed"_split_label0"
    # cp "train_setB_paired_AB_acc_"$seed"_split_label0"/* "train_all_paired_AB_acc_"$seed"_split_label0"
    # cd ..

    python3 main_mmbind_3_fuse_contrastive.py --seed $seed --common_modality acc --gpu 3 --dataset_split split_label0
    python3 main_mmbind_3_incomplete_contrastive.py --seed $seed --common_modality acc --gpu 3 --dataset_split split_label0
done

cd ../evaluation
bash eval.sh split_label0
bash eval_mmbind_incomplete.sh split_label0
cd ../train