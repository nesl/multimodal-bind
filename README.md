# Multimodal-Bind - UTD dataset
Multimodal Embedding Learning with Distributed Incomplete Data

## Processed UTD dataset for skeleton bind
[https://drive.google.com/drive/folders/1gqVcOg5tagfdNNQ9d81oV0AReh--p_O1?usp=sharing](https://drive.google.com/file/d/136zY2L2krizCKf1MPEXY4AwJ09oylty4/view?usp=sharing)

## Code of UTD dataset
#train
Baseline 1: Single-modal autoencoder
Baseline 2: Incomplete multimodal
Baseline 3: Incomplete multimodal + masked vector
Baseline 4: Cross-modal generation (also three steps)
Baseline 5: Dual contrastive
Upper Bound (all natually paired data): Single-modal autoencoder + multimodal contrastive (data of two paired modalities)
mmbind_contarstive (manually paired data by mmbind): Single-modal autoencoder + multimodal contrastive (data of two paired modalities)
mmbind_incomplete_contarstive (origianl data + manually paired data by mmbind): Incomplete multimodal contrastive (data of three incomplete modalities)

#evaluate
Lower_bound: Limited labeled paired data for supervised model finetuening
Baseline 1: Single-modal autoencoder
Baseline 2: Incomplete multimodal
Baseline 3: Incomplete multimodal + masked vector
Baseline 4: Cross-modal generation
Baseline 5: Dual contrastive
Upper Bound (all natually paired data): Single-modal autoencoder + multimodal contrastive
mmbind_contarstive (manually paired data by mmbind): Single-modal autoencoder + multimodal contrastive (data of two paired modalities)
mmbind_incomplete_contarstive (origianl data + manually paired data by mmbind): Incomplete multimodal contrastive (data of three incomplete modalities)
