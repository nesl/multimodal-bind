# MMFI Dataset Evaluations

## Downloading and Processing the MMFI Dataset
Refer to the [MMFI Webpage] (https://ntu-aiot-lab.github.io/mm-fi) for instructions to download the MMFi Dataset
Process the data by placing all the data for a particular sample (subject performing an action) into a single `.pickle` file with the keys following the format in `./shared_files/PickleDataset.py`.

## Running evaluations on the MMFI Dataset

Please refer to the following subfolders for instructions on how to run the training and evaluation
- `MMFI_Code_2_FT`: Evaluations after finetuning on two subjects
- `MMFI_Code_4_FT`: Evaluations after finetuning on four subjects
- `MMFI_Code_8_FT`: Evaluations after finetuning on eight subjects
- `MMFI_Code_Diff_Envs`: Evaluations the impact of domain shift across pretrain and finetune
- `MMFI_Code_Label_Binding`: Evaluate label binding results
- `MMFI_Code_Random_Binding`: Randomly generate the paired dataset for MMBind
- `MMFI_Code_Weighted`: The weighted contrastive learning results and baselines that we show in Table 3
- `MMFI_Code_WiFi_Binding`: Performing data binding with the non-descriptive WiFi modality

