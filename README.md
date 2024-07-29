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
- MMBind (origianl data + manually paired data by mmbind): Incomplete multimodal contrastive (data of three incomplete modalities)
- Upper Bound (all natually paired data): multimodal supervised learning, incomplete data for three-modaliy modal (data of two paired modalities)
