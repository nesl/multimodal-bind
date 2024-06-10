#!/bin/bash
python3 main_baseline2_mask_contrastive.py --seed $1 --common_modality acc --gpu 3
python3 main_baseline2_mask_contrastive.py --seed $1 --common_modality gyro --gpu 3
python3 main_baseline2_mask_contrastive.py --seed $1 --common_modality mag --gpu 3

cd ../evaluation
bash eval_baseline2.sh $1