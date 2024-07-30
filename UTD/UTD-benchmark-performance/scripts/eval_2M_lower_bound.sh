#!/bin/bash

cd ..

python3 main_2M_lower_bound.py --modality acc_gyro --dataset train_C/label_216/
python3 main_2M_lower_bound.py --modality skeleton_gyro --dataset train_C/label_216/
python3 main_2M_lower_bound.py --modality acc_skeleton --dataset train_C/label_216/

python3 main_2M_lower_bound.py --modality acc_gyro --dataset train_C/label_162/
python3 main_2M_lower_bound.py --modality skeleton_gyro --dataset train_C/label_162/
python3 main_2M_lower_bound.py --modality acc_skeleton --dataset train_C/label_162/

python3 main_2M_lower_bound.py --modality acc_gyro --dataset train_C/label_108/
python3 main_2M_lower_bound.py --modality skeleton_gyro --dataset train_C/label_108/
python3 main_2M_lower_bound.py --modality acc_skeleton --dataset train_C/label_108/

python3 main_2M_lower_bound.py --modality acc_gyro --dataset train_C/label_54/
python3 main_2M_lower_bound.py --modality skeleton_gyro --dataset train_C/label_54/
python3 main_2M_lower_bound.py --modality acc_skeleton --dataset train_C/label_54/