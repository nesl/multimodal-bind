#!/bin/bash
cd train
python3 label_baseline3_supervised.py --seed $1 
cd ../evaluation
python3 label_baseline3_evaluate.py --seed $1 
mv save_baseline3/results save_baseline3/results_${1}
