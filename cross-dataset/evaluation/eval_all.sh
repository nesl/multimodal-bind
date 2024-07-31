#!/bin/bash

cd ..

python3 main_fuse_sup_lower_bound.py --train_ratio 0.1
python3 main_fuse_sup_lower_bound.py --train_ratio 0.05
python3 main_fuse_sup_lower_bound.py --train_ratio 0.01

python3 main_fuse_acc_pair.py --train_ratio 0.1
python3 main_fuse_acc_pair.py --train_ratio 0.05
python3 main_fuse_acc_pair.py --train_ratio 0.01

python3 main_fuse_contrastive_label_pair.py --train_ratio 0.1
python3 main_fuse_contrastive_label_pair.py --train_ratio 0.05
python3 main_fuse_contrastive_label_pair.py --train_ratio 0.01