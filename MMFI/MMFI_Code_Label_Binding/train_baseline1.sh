#!/bin/bash
cd train
python3 label_baseline1_supervised.py --seed $1 --modality depth --train_config ../Configs/config_train_A.yaml 
python3 label_baseline1_supervised.py --seed $1 --modality mmwave --train_config ../Configs/config_train_B.yaml
cd ../evaluation
python3 label_baseline1_evaluate.py --seed $1
mv save_unimodal/results save_unimodal/results_${1}
