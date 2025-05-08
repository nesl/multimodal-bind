cd train

python3 baseline1.py
python3 baseline2.py
python3 baseline3.py

python3 main_mmbind_1_generate_dataset.py
python3 main_mmbind_2_fuse_contrastive.py
python3 main_mmbind_3_supervised.py

python3 upper_bound_1_2M_incomplete_contrastive.py
python3 upper_bound_2_supervised.py


cd evaluate
python3 baseline1.py
python3 baseline2.py
python3 baseline3.py
python3 lower_bound.py
python3 mmbind_incomplete.py
python3 upper_bound.py




