In this folder, we provide the results for **label binding** with MMFI. 

The .sh bash files correspond to each of the baselines shown in Table 4. Although they are titled train_X., they perform both training and evaluation
- train_baseline1.sh: Corresponds to the **Unimodal** baseline
- train_baseline2.sh: Corresponds to the **MIM** baseline
- train_baseline3.sh: Corresponds to the **MPM** baseline
- train_mmbind.sh: The **MMBind** method
- train_upperbound.sh: Provides the **upper bound** performance with naturally paired data

Running these scripts will provide a directory in `./train` where the models are stored, and a correspoding directory in `./evaluation` where the results are stored. 