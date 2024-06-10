#!/bin/bash
python3 main_baseline2_mask_contrastive.py --seed 123 --common_modality acc --gpu 3
python3 main_baseline2_mask_contrastive.py --seed 123 --common_modality gyro --gpu 3
python3 main_baseline2_mask_contrastive.py --seed 123 --common_modality mag --gpu 3
