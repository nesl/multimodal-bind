# Multimodal-Bind - UTD dataset Label Bind
Multimodal Embedding Learning with Distributed Incomplete Data

## Processed UTD dataset for label bind
[https://drive.google.com/drive/folders/1gqVcOg5tagfdNNQ9d81oV0AReh--p_O1?usp=sharing](https://drive.google.com/file/d/136zY2L2krizCKf1MPEXY4AwJ09oylty4/view?usp=sharing)

## Code of UTD dataset

# No Paired Dataset (train_C)
mmbind_contarstive_supervise has the best performance

#train_and_evaluate
- Lower_bound: N/A as we did not add paired labeled data train_C
- Baseline 1: Unimodal supervised learning 
- Baseline 2: Incomplete multimodal supervised learning, no contrastive as no paired data
- Baseline 3: Incomplete multimodal supervised learning + prompt
- Baseline 4: N/A as we did not add paired labeled data train_C (Cross-modal generation)
- Baseline 5: N/A as we did not add paired labeled data train_C, (Dual contrastive)
- Upper Bound (all natually paired data from train_A and train_B): multimodal supervised learning
- MMbind:
  * Step 1: Pair data according to labels
  * Step 2:
    * mmbind_contarstive_supervise (manually paired data by mmbind): multimodal contrastive learning + supervised learning (data of two paired modalities)
    * mmbind_incomplete_contarstive_supervise (origianl data + manually paired data by mmbind): incomplete multimodal contrastive  + supervised learning (data of incomplete modalities + data of two paired modalities)

# Having Paired Dataset (train_C)
mmbind_contarstive_supervise has the best performance

#train
- Baseline 1: Single-modal autoencoder
- Baseline 2: Incomplete multimodal
- Baseline 3: Incomplete multimodal + masked vector
- Baseline 4: Cross-modal generation (also three steps)
- Baseline 5: Dual contrastive
- Upper Bound (all natually paired data): Single-modal autoencoder + multimodal contrastive (data of two paired modalities)
- mmbind_contarstive (manually paired data by mmbind): Single-modal autoencoder + multimodal contrastive (data of two paired modalities)
- mmbind_incomplete_contarstive (origianl data + manually paired data by mmbind): Incomplete multimodal contrastive (data of three incomplete modalities)

#evaluate
- Lower_bound: Limited labeled paired data for supervised model finetuening
- Baseline 1: Single-modal autoencoder
- Baseline 2: Incomplete multimodal
- Baseline 3: Incomplete multimodal + masked vector
- Baseline 4: Cross-modal generation
- Baseline 5: Dual contrastive
- Upper Bound (all natually paired data): Single-modal autoencoder + multimodal contrastive
- mmbind_contarstive (manually paired data by mmbind): Single-modal autoencoder + multimodal contrastive (data of two paired modalities)
- mmbind_incomplete_contarstive (origianl data + manually paired data by mmbind): Incomplete multimodal contrastive (data of three incomplete modalities)
