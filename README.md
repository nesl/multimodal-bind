## MMFI-MMBIND

### Key Folders
Top Level: 

**MMFI Code**: Contains all code for this project
   
**MMFI_Dataset**: Base dataset containing all the pickle files under their respective Environment/Subject/Action folders
* **Configs**: This folder holds all the configuration files that are used to generate the dataset. Configuration files specify the subjects/actions + modalities
* **evaluation**: Used for fine-tuning/training models from scratch
* **train**: Contains all the pretraining files

**MMFI_Dataset_train_depth...**: This is where the new paired pickle files will be generated after performing `main_mmbind_2_measure_similarity.py` w.r.t depth

**MMFI_Dataset_train_mmwave...**: This is where the new paired pickle files will be generated after performing `main_mmbind_2_measure_similarity.py` w.r.t mmwave


After running the skeleton pairing for both depth and mmwave make sure to manually concat the datasets if you want a combined dataset for step 3

python3 main_mmbind_3_fuse_contrastive.py --load_pretrain="no_pretrain"

