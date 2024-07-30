#!/bin/bash

cd ..

python3 main_mmbind_1_label_similarity.py
python3 main_mmbind_2_pair_data.py
python3 main_mmbind_3_incomplete_contarstive_supervise.py