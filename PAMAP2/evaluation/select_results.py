# import scipy.io as scio
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import shutil
# import random
# import os

# random.seed(4)


# settings = []

# label_set = 'label_54'
# directory = './save_train_C/{}/'.format(label_set)
# folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

# # print(folders)

# for folder_id in range(len(folders)):
# 	if 'upper_load' in folders[folder_id]:
# 		#print(folders[folder_id])
# 		if 'contrative' in folders[folder_id]:
# 			load_path = directory + folders[folder_id] + '/model_pretrain_AE/results/'
# 		else: 
# 			load_path = directory + folders[folder_id] + '/results/'
# 		print(load_path)

# 		files = os.listdir(load_path)

# 		save_path = './upper_load_pretrain_results-0425/{}/{}/'.format(label_set, folders[folder_id])
# 		print(save_path)

# 		if not os.path.isdir(save_path):
# 		    os.makedirs(save_path)


# 		# Iterate through each file and copy it to the destination directory
# 		for file in files:
# 		    source_file = os.path.join(load_path, file)
# 		    destination_file = os.path.join(save_path, file)
# 		    shutil.copy(source_file, destination_file)


