# #!/bin/bash
# cd train
# python3 main_mmbind_1_acc_autencoder.py --seed $1
# python3 main_mmbind_2_measure_similarity.py --seed $1
# python3 main_mmbind_2_measure_similarity.py --seed $1 --reference_modality mag
# cd save_mmbind
# mkdir train_all_paired_AB
# cp train_gyro_paired_AB/* train_all_paired_AB
# cp train_mag_paired_AB/* train_all_paired_AB
# cd ..


# python3 main_mmbind_3_fuse_contrastive.py --seed $1

# cd ../evaluation
# python3 main_fuse_sup_mmbind.py --seed $1
# cd save_train_C/label_216/mmbind/results
# mv test_accuracy.txt test_accuracy_${1}.txt
# cd /home/jason/Documents/MMBind_PAMAP/PAMAP-acc-bind/train/save_mmbind
# rm -r train*


# python3 main_mmbind_2_measure_similarity.py --common_modality gyro --reference_modality setA
# python3 main_mmbind_2_measure_similarity.py --common_modality gyro --reference_modality setB

# python3 main_mmbind_2_measure_similarity.py --common_modality acc --reference_modality setA
# python3 main_mmbind_2_measure_similarity.py --common_modality acc --reference_modality setB

# python3 main_mmbind_2_measure_similarity.py --common_modality mag --reference_modality setA
# python3 main_mmbind_2_measure_similarity.py --common_modality mag --reference_modality setB

mkdir train_all_paired_AB_acc
cp train_setA_paired_AB_acc/* train_all_paired_AB_acc
cp train_setB_paired_AB_acc/* train_all_paired_AB_acc

mkdir train_all_paired_AB_gyros
cp train_setA_paired_AB_gyros/* train_all_paired_AB_gyros
cp train_setB_paired_AB_gyros/* train_all_paired_AB_gyros

mkdir train_all_paired_AB_mag
cp train_setA_paired_AB_mag/* train_all_paired_AB_mag
cp train_setB_paired_AB_mag/* train_all_paired_AB_mag
