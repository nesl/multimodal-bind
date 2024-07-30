#!/bin/bash

cd ..

python3 main_supfuse_baseline1_unimodal_supervise.py --dataset train_C/label_216/

python3 main_supfuse_baseline1_unimodal_supervise.py --dataset train_C/label_162/

python3 main_supfuse_baseline1_unimodal_supervise.py --dataset train_C/label_108/

python3 main_supfuse_baseline1_unimodal_supervise.py --dataset train_C/label_54/
