#!/bin/bash

python3 main_fuse_sup_lower_bound.py --seed 123 --common_modality acc --gpu 3
python3 main_fuse_sup_lower_bound.py --seed 123 --common_modality gyro --gpu 3
python3 main_fuse_sup_lower_bound.py --seed 123 --common_modality mag --gpu 3
