#!/bin/bash

cd ..

python3 main_single_lower_bound.py --modality acc --dataset train_C/label_216/
python3 main_single_lower_bound.py --modality gyro --dataset train_C/label_216/
python3 main_single_lower_bound.py --modality skeleton --dataset train_C/label_216/

python3 main_single_lower_bound.py --modality acc --dataset train_C/label_162/
python3 main_single_lower_bound.py --modality gyro --dataset train_C/label_162/
python3 main_single_lower_bound.py --modality skeleton --dataset train_C/label_162/

python3 main_single_lower_bound.py --modality acc --dataset train_C/label_108/
python3 main_single_lower_bound.py --modality gyro --dataset train_C/label_108/
python3 main_single_lower_bound.py --modality skeleton --dataset train_C/label_108/

python3 main_single_lower_bound.py --modality acc --dataset train_C/label_54/
python3 main_single_lower_bound.py --modality gyro --dataset train_C/label_54/
python3 main_single_lower_bound.py --modality skeleton --dataset train_C/label_54/