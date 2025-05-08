In this folder, we assume that we only have labeled data from **ten** subjects, but the subjects are grouped in a manner such that the subject utilized for pretraining are from different environments than those used in fine-tuning. We use this to investigate effect of domain shift among pretraining and fine-tuning data. This labeled data will be used for fine-tuning after MMFI Pretraining

The .sh bash files correspond to each of the baselines present in 6.1.2. Although they are titled train_X., they perform both training and evaluation
- train_baseline1.sh: Corresponds to the **Unimodal** baseline
- train_baseline2.sh: Corresponds to the **MIM** baseline
- train_baseline3.sh: Corresponds to the **MPM** baseline
- train_baseline4.sh: Corresponds to the **CMG** baseline
- train_dual_contrastive.sh: Corresponds to the **DCM** baseline
- train_lowerbound.sh: Corresponds to **Lower Bound** baseline
- train_mmbind_incomplete.sh: The **MMBind** method
- train_upperbound.sh: Provides the **upper bound** performance with naturally paired data

Running these scripts will provide a directory in `./train` where the models are stored, and a correspoding directory in `./evaluation` where the results are stored. 