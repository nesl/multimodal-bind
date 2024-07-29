# MMBind - UTD dataset Acc Bind
MMBind on UTD dataset, where the binding modality is the accelerometer data.

## Processed UTD dataset for acc bind (Same as label bind)
[https://drive.google.com/file/d/1xossSJyJZ51rB5xLvHsVaqgTKn5nmsQE/view?usp=sharing](https://drive.google.com/file/d/1xossSJyJZ51rB5xLvHsVaqgTKn5nmsQE/view?usp=sharing)

## Structure of Code
**mmbind_incomplete_contarstive** has the best performance, without loading pretrained single-modal autoencoder.

#Train
- Baseline 1: Single-modal autoencoder
- Baseline 2: Incomplete multimodal
- Baseline 3: Incomplete multimodal + masked vector
- Baseline 4: Cross-modal generation (also three steps)
- Baseline 5: Dual contrastive
- mmbind_contarstive (manually paired data by mmbind): Single-modal autoencoder + multimodal contrastive (data of two paired modalities)
- **mmbind_incomplete_contarstive** (origianl data + manually paired data by mmbind): Incomplete multimodal contrastive (data of three incomplete modalities)
- Upper Bound (all natually paired data):
  * Two modality fusion (main_upper_bound_2_fuse_contrastive.py): single-modal autoencoder + multimodal contrastive (data of two paired modalities)
  * **Three modality fusion** (main_upper_bound_2_3M_contrastive.py): single-modal autoencoder + three multimodal contrastive (data of three paired modalities)

#Evaluate
- Lower_bound: Limited labeled paired data for supervised model finetuening
- Baseline 1: Single-modal autoencoder
- Baseline 2: Incomplete multimodal
- Baseline 3: Incomplete multimodal + masked vector
- Baseline 4: Cross-modal generation
- Baseline 5: Dual contrastive
- mmbind_contarstive (manually paired data by mmbind): Single-modal autoencoder + multimodal contrastive (data of two paired modalities)
- mmbind_incomplete_contarstive (origianl data + manually paired data by mmbind): Incomplete multimodal contrastive (data of three incomplete modalities)
- Upper Bound (all natually paired data): 
  * Two modality fusion (main_fuse_sup_upper_bound.py): multimodal supervised learning (data of two paired modalities)
  * **Three modality fusion** (main_fuse_sup_upper_bound_3M.py): multimodal supervised learning, incomplete data for three-modaliy modal (data of two paired modalities)
