
## Download instructions

Download the processed PAMAP dataset at this link: https://drive.google.com/file/d/1WaNM3fGJ8VEsCBe3weRXceDRGjokiph6/view?usp=sharing

Place it in the same directory as this cloned repository 

i.e: 

parent directory

            PAMAP-Code
            
            PAMAP_Dataset
            
The folder should be named as PAMAP_Dataset, and its direct children directories should be trainA, trainB, trainC, and test


## Data Format

Currently, the first number in the filename represents the action performed. The second number is simply an index to prevent duplicate files, and can be disregarded.

The each data sample has shape 1000 x 54 (100 Hz sampling rate for 10 seconds), and the 54 fields include other data + imu, gyro, mag. Note that this only wrist IMU data

Train Dataset A is treated as an unlabeled Acc + Gyro dataset

Train Dataset B is treated as an unlabeled Acc + Mag dataset

Train Dataset C is treated as a labeled Gyro + Mag dataset


## Steps for Running

### Baseline 1: Autoencoders

**Training**

We train independent autoencoders (gyro on dataset A, and mag on dataset B), and load the pretrained weights of each encoder to finetune on dataset C

```
python3 main_baseline1_separate_autoencoder.py // Trains the gyro encoder
python3 main_baseline2_separate_autoencoder.py --dataset train_B // Trains the mag encoder
```

This will generate two folders under `train/save_baseline1` where the weights are stored

**Evaluation**

We evaluate by running the following code `python3 main_fuse_sup_baseline1.py`, which will automatically load the pretrained weights and fine-tune on dataset C, while also performing validation on the test dataset

Results are saved under `evaluation/save_train_C/label_216/baseline1/results/test_accuracy.txt`


### Baseline 2: Incomplete Multimodal Training

In this baseline, we utilize three encoders to generate three embeddings (one for each modality), and then employ contrastive pre-training. We use our incomplete modality datasets (trainA, trainB) by replacing the missing modality with zeros. Thus, per data sample, two of the three embeddings will be real data, while the other is replaced with zero. 

**Training**

We can either train the model purely from scratch `python3 main_baseline2_mask_contrastive.py`, or with the pretrained encoders from the previous step `python3 main_baseline2_mask_contrastive.py --load_pretrain`

This saves the model weights under the directory `./save_baseline2/`

**Evaluation** 

We evaluate with the code `python3 main_fuse_sup_baseline2.py`, with results saved under `evaluation/save_train_C/label_216/baseline2/results/test_accuracy.txt`

### Baseline 3: Dual Contrastive

We establish an additional baseline referred to as Dual Contrastive, which expands upon the method in baseline 2. We would provide zero input for missing modalities, and generate three latent representations for our three modalities, after which we employ contastive loss across *all three embeddings*, even the zero data latent representation. In dual-contrastive learning, we throw out the latent corresponding to the missing modlaity, and simply calculate paired contrastive loss. For a given sample, we will only utilize two of the three encoders. The idea is that if we train a good shared modality encoder, then similar data samples will be implicitly paired through the shared modality

**Training**

`python3 main_dualcontrastive_1_train_contrastive.py` from scratch, `python3 main_dualcontrastive_1_train_contrastive.py --load_pretrain` from pretrained baseline1 weights
Saves under `./save_dual_contrastive`

**Evaluation**

Evaluate with `python3 main_fuse_sup_dual_contrastive.py`

### Lower Bound

No need for pretraining, we simply train the model from scratch on the finetune dataset C and evaluate

**Evaluation**

Evaluate with `python3 main_fuse_sup_lower_bound.py`

### MMBind

MMBind involves three stages. First, we train one single encoder on unimodal data from the shared modality. For example, if dataset 1 has (A, B) and dataset 2 has (B, C), then we train a unimodal encoder with the aggregated data of B from both datasets 1 and 2. This encoder is pretrained as half of an autoencoder with reconstruction loss. Secondly, we use that autoencoder to generate synthetically paired (A, C) data from datasets 1 and 2. Finally, we train A and C encoders through contrastive pretraining

**Training**

All model weights and new datasets saved under appropriate folder inside directory `./save_mmbind'`
Step 1: `python3 main_mmbind_1_acc_autencoder.py`

Step 2: `python3 main_mmbind_2_measure_similarity.py` will generate a paired dataset with gyroscope as the reference modality (i.e, we take each sample of dataset 1 and go find the closest sample from dataset 2). Similarity `python3 main_mmbind_2_measure_similarity.py --reference_modality mag` does the same thing with magnometer as the reference modality. After performing these two steps, we can aggregate the two by creating a new directory `train_all_paired_AB`, and copying the child folders of `train_gyro_paired_AB` and `train_mag_paired_AB` into there.

Step 3: `python3 main_mmbind_3_fuse_contrastive.py` will by default train on the merged `train_all_paired_AB` dataset. Use the --dataset flag to specify another dataset. This step ends by generating the contrastive weights. By default, it does not load pretrained autoencoder weights, use --load_pretrain to specify that

**Evaluations**

`python3 main_fuse_sup_mmbind.py` will take the contrastive weights that we just pretrained.

## Upper Bound

Instead of using MMBind to generate artifically paired data samples, evaluate the performance of actually paired data samples. So, we simply start at the contrastive pretraining stage.

**Training**

Step 1: `python3 main_upper_bound_1_single_autoencoder.py`. Similar to the baseline1, however, we can specify which dataset we want to do this pretraining on. By default --dataset is train_AB
Step 2: `python3 main_upper_bound_1_fuse_contrastive.py`. Perform contrastive pretraining, loading the pretrained weights

**Evaluation**

`python3 main_fuse_sup_upper_bound.py`







