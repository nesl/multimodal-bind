#!/bin/bash
python3 main_dualcontrastive_1_train_contrastive.py --seed $1 --common_modality acc --gpu 3
python3 main_dualcontrastive_1_train_contrastive.py --seed $1 --common_modality gyro --gpu 3
python3 main_dualcontrastive_1_train_contrastive.py --seed $1 --common_modality mag --gpu 3

cd ../evaluation
bash eval_dual.sh