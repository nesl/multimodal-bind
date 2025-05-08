# cd train
# # python3 main_baseline2_incomplete_contrastive.py
# # python3 main_baseline3_vector2_attach_incomplete_contrastive.py
# python3 main_mmbind_1_img_autoencoder.py
# python3 main_mmbind_2_measure_similarity.py
# python3 main_mmbind_2_measure_similarity.py --reference_modality depth
# python3 main_mmbind_3_incomplete_contrastive.py
# python3 main_upper_bound_2_3M_incomplete_contrastive.py


cd evaluate
python3 baseline1.py
python3 baseline2.py
python3 baseline4.py
python3 baseline5.py
python3 imagebind.py

