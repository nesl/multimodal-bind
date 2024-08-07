#!/bin/bash
cd train
python3 label_baseline2_supervised.py --seed $1 
cd ../evaluation
python3 label_baseline2_evaluate.py --seed $1 
mv save_baseline2/results save_baseline2/results_${1}
