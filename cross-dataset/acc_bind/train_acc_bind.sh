#!/bin/bash

cd ..

python3 main_mmbind_1_acc_AE.py

python3 main_mmbind_2_pair.py --reference_modality mag
python3 main_mmbind_2_pair.py --reference_modality gyro

python3 main_mmbind_3_contrastive.py