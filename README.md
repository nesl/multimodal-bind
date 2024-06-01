
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
