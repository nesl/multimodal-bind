MMFI_Code_X_FT: Incomplete Contrastive Learning with noise, but no weighted contrastive

MMFI_Code_Diff_Envs: Train A and Train B come from distinct environments, C and test same environment. No weighted contrastive either

MMFI_Code_Label_Bind: Label binding without weighted contrastive loss

MMFI_Code_Random_Binding: Tests the effectiveness of using randomly generated pairs

MMFI_Code_Weighted: Performs weighted contrastive loss, assign 0.1 to the naturally paired, incomplete samples, and use the similarity derived from the pairing step

MMFI_Code_Wifi_Binding: Generate pairs based off WiFi similarity instead, no weighted contrastive


