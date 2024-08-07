#!/bin/bash
cd train
python3 label_upperbound_1_contrastive.py --seed $1 
python3 label_upperbound_2_supervised.py --seed $1 
cd ../evaluation
python3 label_upperbound_evaluate.py --seed $1 
mv save_upperbound/results save_upperbound/results_${1}

