#!/bin/bash

python3 main_fuse_sup_baseline1.py --seed $1 --common_modality acc
python3 main_fuse_sup_baseline1.py --seed $1 --common_modality gyro
python3 main_fuse_sup_baseline1.py --seed $1 --common_modality mag
