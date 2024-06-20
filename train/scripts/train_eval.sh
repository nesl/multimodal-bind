#!/bin/bash

cd ..

cd ../evaluation
bash eval_baseline1.sh split_label0
cd ../train

cd ../evaluation
bash eval_baseline2.sh split_label0
cd ../train

cd ../evaluation
bash eval_dual.sh split_label0
cd ../train