#!/bin/bash

cd ..

python3 main_mmbind_1_label_similarity.py
python3 main_mmbind_2_pair_data.py

python3 main_mmbind_3_incomplete_contarstive_supervise.py --dataset train_C/label_216/

python3 main_mmbind_3_incomplete_contarstive_supervise.py --dataset train_C/label_162/

python3 main_mmbind_3_incomplete_contarstive_supervise.py --dataset train_C/label_108/

python3 main_mmbind_3_incomplete_contarstive_supervise.py --dataset train_C/label_54/