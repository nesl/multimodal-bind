# MMBind - UTD dataset
MMBind on the UTD dataset, where the binding modality is the accelerometer data or labels.

## Processed UTD dataset
[https://drive.google.com/file/d/1xossSJyJZ51rB5xLvHsVaqgTKn5nmsQE/view?usp=sharing](https://drive.google.com/file/d/1xossSJyJZ51rB5xLvHsVaqgTKn5nmsQE/view?usp=sharing)

- Same data splitting for Acc-bind and Label-bind.
- Unzipped and put the data under the folder "./UTD/"

## Structure of Code: Acc binding
#Train (Train_A + Train_B)
- Baseline 1: Single-modal autoencoder
- Baseline 2: Incomplete multimodal
- Baseline 3: Incomplete multimodal + masked vector
- Baseline 4: Cross-modal generation (also three steps)
- Baseline 5: Dual contrastive
- **MMBind** (original data + manually paired data by mmbind): Incomplete multimodal contrastive (data of three incomplete modalities)
- Upper Bound (all naturally paired data): single-modal autoencoder + three multimodal contrastive (data of three paired modalities)

#Evaluate (Finetune with Train_C)
- Lower_bound: Limited labeled paired data for supervised model finetuning
- Baseline 1: Single-modal autoencoder
- Baseline 2: Incomplete multimodal
- Baseline 3: Incomplete multimodal + masked vector
- Baseline 4: Cross-modal generation
- Baseline 5: Dual contrastive
- **MMBind** (original data + manually paired data by mmbind): Incomplete multimodal contrastive (data of three incomplete modalities)
- Upper Bound (all naturally paired data): multimodal supervised learning, incomplete data for three-modality modal (data of two paired modalities)

## Structure of Code: Label binding

## (1) All Class Overlap

#Train (Train_A + Train_B)
- **Baseline 1**: Unimodal supervised learning 
- **Baseline 2**: Incomplete multimodal supervised learning, no contrastive as no paired data
- **Baseline 3**: Incomplete multimodal supervised learning + prompt
- **MMbind**:
  * Step 1: Pair data according to labels: Generate as much as possible paired multimodal samples for each unimodal sample, e.g., N = N_A * N_B_i + N_B * N_A_i (N_B_i is the # of samples for class i)
  * Step 2: (manually paired data by mmbind): multimodal contrastive learning + supervised learning (data of two paired modalities)

#Evaluate (Finetune with Train_C)
- **Lower_bound**: multimodal supervised learning with train_C
- **Baseline 1**: multimodal supervised finetuning with train_C
- **Baseline 2**: multimodal supervised finetuning with train_C
- **Baseline 3**: multimodal supervised finetuning with train_C
- **Upper Bound** (all naturally paired data from train_A+train_B+train_C): multimodal contrastive + supervised learning with train_C
- **MMbind_contarstive_supervise** (manually paired data by mmbind): multimodal contrastive learning + supervised learning with train_C

## (2) Partial Class Overlap
#Train (Train_A + Train_B) and evaluate
- Lower_bound: N/A as we did not add paired labeled data train_C
- **Baseline 1**: Unimodal supervised learning
- **Baseline 2**: Incomplete multimodal supervised learning, no contrastive as no paired data
- **Baseline 3**: Incomplete multimodal supervised learning + prompt
- Baseline 4: N/A as we did not add paired labeled data train_C (Cross-modal generation)
- Baseline 5: N/A as we did not add paired labeled data train_C, (Dual contrastive)
- **Upper Bound** (all naturally paired data from train_A and train_B): multimodal contrastive + supervised learning

- **MMbind**:
  * Step 1: Pair data according to labels. Generate as much as possible paired multimodal samples for each unimodal sample, e.g., N = N_A * N_B_i + N_B * N_A_i (N_B_i is the # of samples for class i)
  * Step 2: multimodal contrastive learning + supervised learning (data of two paired modalities)

## Usage

Scripts are available for sensor binding and label binding experiments. See `scripts/` in each subfolder.
