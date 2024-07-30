#!/bin/bash

cd ..

python3 main_baseline1_unimodal_supervise.py --dataset train_A --modality skeleton --dataset train_C/label_216/
python3 main_baseline1_unimodal_supervise.py --dataset train_B --modality gyro --dataset train_C/label_216/

python3 main_baseline1_unimodal_supervise.py --dataset train_A --modality skeleton --dataset train_C/label_162/
python3 main_baseline1_unimodal_supervise.py --dataset train_B --modality gyro --dataset train_C/label_162/

python3 main_baseline1_unimodal_supervise.py --dataset train_A --modality skeleton --dataset train_C/label_108/
python3 main_baseline1_unimodal_supervise.py --dataset train_B --modality gyro --dataset train_C/label_108/

python3 main_baseline1_unimodal_supervise.py --dataset train_A --modality skeleton --dataset train_C/label_54/
python3 main_baseline1_unimodal_supervise.py --dataset train_B --modality gyro --dataset train_C/label_54/