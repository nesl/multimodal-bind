# MMBind - UTD dataset Acc Bind
MMBind on UTD dataset, where the binding modality is the accelerometer data.

## Processed UTD dataset for acc bind (Same as label bind)
[https://drive.google.com/file/d/1xossSJyJZ51rB5xLvHsVaqgTKn5nmsQE/view?usp=sharing](https://drive.google.com/file/d/1xossSJyJZ51rB5xLvHsVaqgTKn5nmsQE/view?usp=sharing)

## Structure of Code
#Train
- Baseline 1: Single-modal autoencoder
- Baseline 2: Incomplete multimodal
- Baseline 3: Incomplete multimodal + masked vector
- Baseline 4: Cross-modal generation (also three steps)
- Baseline 5: Dual contrastive
- **MMBind** (origianl data + manually paired data by mmbind): Incomplete multimodal contrastive (data of three incomplete modalities)
- Upper Bound (all natually paired data): single-modal autoencoder + three multimodal contrastive (data of three paired modalities)

#Evaluate
- Lower_bound: Limited labeled paired data for supervised model finetuening
- Baseline 1: Single-modal autoencoder
- Baseline 2: Incomplete multimodal
- Baseline 3: Incomplete multimodal + masked vector
- Baseline 4: Cross-modal generation
- Baseline 5: Dual contrastive
- **MMBind** (origianl data + manually paired data by mmbind): Incomplete multimodal contrastive (data of three incomplete modalities)
- Upper Bound (all natually paired data): multimodal supervised learning, incomplete data for three-modaliy modal (data of two paired modalities)

## Usage

Scripts are available for sensor binding and label binding experiments. See `scripts/`.

Overall, the common options used are
```
python3 XXX.py --learning_rate XXX
```

**Description of each option**
- `--seed`: the seed of the experiment, used 41-45
- `--gpu`: Specify which GPU to use (single GPU only)
- `--dataset_split`: Specify the dataset split (or the index folder) to use, explore different splitting methods
- `--common_modality`: The binding modality, explore different modality used as the binding mod. For baselines which do not use the binding modality, the reference modalities (modalities that do not have pairs) are automatically selected with the binding modality. 
