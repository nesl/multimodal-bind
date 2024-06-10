#!/bin/bash

python3 main_fuse_sup_mmbind.py --seed $1 --common_modality acc --learning_rate 1e-4 --weight_decay 1e-4
python3 main_fuse_sup_mmbind.py --seed $1 --common_modality gyro --learning_rate 1e-4 --weight_decay 1e-4
python3 main_fuse_sup_mmbind.py --seed $1 --common_modality mag --learning_rate 1e-4 --weight_decay 1e-4
