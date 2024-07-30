#!/bin/bash
cd ..


python3 main_mmbind_1_acc_autencoder.py

python3 main_mmbind_2_measure_similarity.py --reference_modality skeleton
python3 main_mmbind_2_measure_similarity.py --reference_modality gyro

python3 main_mmbind_3_incomplete_contrastive.py
