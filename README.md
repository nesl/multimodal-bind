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