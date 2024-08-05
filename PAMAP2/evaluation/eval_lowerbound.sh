# #!/bin/bash


#!/bin/bash
for seed in 41 42 43 44 45;
do
    python3 main_fuse_sup_lower_bound_abc.py --seed $seed --common_modality acc --gpu 3 --dataset_split $1 --learning_rate 0.00005
    python3 main_fuse_sup_lower_bound.py --seed $seed --common_modality acc --gpu 3 --dataset_split $1 --learning_rate 0.00005
done
