#!/bin/bash

cd ..

python3 main_baseline1_unimodal_supervise.py --dataset train_A --modality skeleton
python3 main_baseline1_unimodal_supervise.py --dataset train_B --modality gyro