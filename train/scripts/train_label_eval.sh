#!/bin/bash

cd ..


for seed in 42 43 44 45;
do
    python3 main_baseline1_label_unimodal_supervise.py --gpu 3 --seed $seed --common_modality acc --dataset_split split_0 --pairing True
    python3 main_baseline2_label_incomplete_multimodal.py --gpu 3 --seed $seed --common_modality acc --dataset_split split_0 --pairing True
    python3 main_baseline3_label_vector_attach_incomplete_multimodal.py --gpu 3 --seed $seed --common_modality acc --dataset_split split_0 --pairing True
    python3 main_mmbind_label_2_contrastive_supervise.py --gpu 3 --seed $seed --common_modality acc --dataset_split split_0 --pairing True --num_class 7
    python3 main_mmbind_label_2_incomplete_contrastive_supervise.py --gpu 3 --seed $seed --common_modality acc --dataset_split split_0 --pairing True --num_class 7
    python3 main_mmbind_label_2_contrastive_supervise.py --gpu 3 --seed $seed --common_modality acc --dataset_split split_0 --pairing True --use_pair True --num_class 7
    python3 main_mmbind_label_2_incomplete_contrastive_supervise.py --gpu 3 --seed $seed --common_modality acc --dataset_split split_0 --pairing True --use_pair True --num_class 7
    python3 main_upper_bound_label.py --gpu 3 --seed $seed --common_modality acc --dataset_split split_0 --pairing True
    python3 main_upper_bound_label_contrastive_supervise.py --gpu 3 --seed $seed --common_modality acc --dataset_split split_0 --pairing True
done
