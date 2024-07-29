# MMBind-PAMAP2 


## Data preprocessing

1. Process .npy file from raw zip file into `./processe/data`, or download the processed file at this ***[link](https://drive.google.com/file/d/1WaNM3fGJ8VEsCBe3weRXceDRGjokiph6/view?usp=sharing)***

2. Create corresponding index file (trainA, trainB, trainC, test)
    ```bash
    cd preprocess
    python generate_index.py # sensor binding
    python generate_index_label.py # label binding
    ```
3. Update the data path if necessary
    - If the `processed_data` and `indices are is in the same directory as the repo root, then no action needed.
    - Else, please update `opt.processed_data_path` and `opt.indice_file` in `modules/option_utils.py` under `train` and `evaluation`.


**Example repo directories**
```bash
multimodal-bind
├── evaluation
│   └── ...
├── indices # indices
│   └── ...
├── preprocess
├── processed_data # processed sensor binding data
├── processed_data_all # processed label binding data
└── train
    └── ...
```

`processed_data` and `processed_data_all` are also available at **[gdrive/PAMAP2](https://drive.google.com/drive/folders/1TMzQ_UOraKF3TYy4bYnTY2xgOqjvPr_E?usp=share_link)**


## Usage

Scripts (the last state being used) are available for sensor binding and label binding experiments. See `scripts/`.

Overall, the common options used are
```
python3 XXX.py --seed XXX --common_modality XXX --gpu XXX --dataset_split XXX --learning_rate XXX --weight_decay XXX
```

**Description of each option**
- `--seed`: the seed of the experiment, used 41-45
- `--gpu`: Specify which GPU to use (single GPU only)
- `--dataset_split`: Specify the dataset split (or the index folder) to use, explore different splitting methods
- `--common_modality`: The binding modality, explore different modality used as the binding mod. For baselines which do not use the binding modality, the reference modalities (modalities that do not have pairs) are automatically selected with the binding modality. 

## Repo structure

```bash
.
├── README.md
├── evaluation
│   ├── eval.sh
│   ├── eval_baseline1.sh
│   ├── eval_baseline2.sh
│   ├── eval_baseline3.sh
│   ├── eval_baseline4.sh
│   ├── eval_dual.sh
│   ├── eval_lowerbound.sh
│   ├── eval_lowerbound_all.sh
│   ├── eval_lowerbound_label.sh
│   ├── eval_mmbind_incomplete.sh
│   ├── eval_mmbind_weighted.sh
│   ├── eval_unimod.sh
│   ├── eval_upperbound.sh
│   ├── main_fuse_sup_baseline1.py
│   ├── main_fuse_sup_baseline2.py
│   ├── main_fuse_sup_baseline3_vector_attach_incomplete_contrastive.py
│   ├── main_fuse_sup_baseline4_contrastive.py
│   ├── main_fuse_sup_dual_contrastive.py
│   ├── main_fuse_sup_lower_bound.py
│   ├── main_fuse_sup_lower_bound_abc.py
│   ├── main_fuse_sup_mmbind.py
│   ├── main_fuse_sup_mmbind_incomplete_contrastive.py
│   ├── main_fuse_sup_mmbind_incomplete_contrastive_weighted.py
│   ├── main_fuse_sup_mmbind_weighted.py
│   ├── main_fuse_sup_upper_bound.py
│   ├── main_sup_allmod.py # supervised all modality
│   ├── main_sup_allmod_abc.py # supervised 
│   ├── main_sup_subsetmod.py
│   ├── main_sup_unimod.py
│   ├── main_supfuse_lower_bound_label.py
│   ├── models
│   ├── modules
│   ├── select_results.py
│   └── shared_files
├── preprocess
│   ├── generate_index.py # generate sensor binding index files
│   ├── generate_index_label.py # generate label binding index files
│   └── process_data.py # process raw data
└── train
    ├── configs # configuration
    ├── main_baseline1_label_unimodal_supervise.py
    ├── main_baseline1_separate_autoencoder.py
    ├── main_baseline2_label_incomplete_multimodal.py
    ├── main_baseline2_mask_contrastive.py
    ├── main_baseline3_label_vector_attach_incomplete_multimodal.py
    ├── main_baseline3_vector_attach_incomplete_contrastive.py
    ├── main_baseline4_1_cross_encoder.py
    ├── main_baseline4_2_cross_generation.py
    ├── main_baseline4_3_fuse_contrastive.py
    ├── main_dualcontrastive_1_train_contrastive.py
    ├── main_mmbind_1_acc_autencoder.py
    ├── main_mmbind_1_unimod_autencoder.py
    ├── main_mmbind_2_measure_similarity.py
    ├── main_mmbind_3_fuse_contrastive.py
    ├── main_mmbind_3_incomplete_contrastive.py
    ├── main_mmbind_3_weighted_fuse_contrastive.py
    ├── main_mmbind_3_weighted_incomplete_contrastive.py
    ├── main_mmbind_label_1_label_pair.py
    ├── main_mmbind_label_1_more_label_pair.py
    ├── main_mmbind_label_2_contrastive_supervise.py
    ├── main_mmbind_label_2_incomplete_contrastive_supervise.py
    ├── main_upper_bound_1_single_autoencoder.py
    ├── main_upper_bound_2_fuse_contrastive.py
    ├── main_upper_bound_2_fuse_contrastive_full.py
    ├── main_upper_bound_2_fuse_incomplete_contrastive.py
    ├── main_upper_bound_label.py
    ├── main_upper_bound_label_contrastive_supervise.py
    ├── models
    ├── modules
    ├── scripts # scripts to run the files
    └── shared_files
```
