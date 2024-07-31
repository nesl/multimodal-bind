import numpy as np
import torch
from sklearn.model_selection import train_test_split
import random



#motionsense data N=12636
#acc: 6.923736999999999 -5.69371 0.23532976051651355 0.6061292876308269 (max, min, mean, std)
#gyro: 11.836725 -14.108059 0.023491750856811232 0.935227478339417
MEAN_OF_IMU_Motionsense = [0.23532976051651355, 0.6061292876308269]
STD_OF_IMU_Motionsense = [0.0234917508568112325, 0.935227478339417]


#shoaib data N=4500
#acc: 19.463 -19.6 -3.197389162046667 5.426273399311965 (max, min, mean, std)
#gyro: 10.004 -9.7653 0.008632669628592595 1.0309785033722876
#mag: 103.68 -89.94 9.58778288888889 25.588296398956754
MEAN_OF_IMU_Shoaib = [-3.197389162046667, 0.008632669628592595, 9.58778288888889]
STD_OF_IMU_Shoaib = [5.426273399311965, 1.0309785033722876, 25.588296398956754]


#realworld data N=21663
#acc: 19.608511 -19.60911 2.4528044603128794 5.5912194524825445 (max, min, mean, std)
#gyro: 10.1170845 -10.174346 0.0073411112346320985 0.9924132095941902
#mag: 166.601 -145.503 -11.10864179091847 27.903385062100536
MEAN_OF_IMU_Realworld = [2.4528044603128794, 0.0073411112346320985, -11.10864179091847]
STD_OF_IMU_Realworld = [5.5912194524825445, 0.9924132095941902, 27.903385062100536]

# random.seed(0)


all_data_folder = "../cross-dataset-processed/"

## load original data
class Multimodal_dataset():
	"""Build dataset from motion sensor data."""
	def __init__(self, x1, x2, x3, y):

		self.data1 = x1.tolist() #concate and tolist
		self.data2 = x2.tolist() #concate and tolist
		self.data3 = x3.tolist() #concate and tolist
		self.labels = y.tolist() #tolist

		self.data1 = torch.tensor(self.data1) # to tensor
		self.data2 = torch.tensor(self.data2) # to tensor
		self.data3 = torch.tensor(self.data3) # to tensor
		self.labels = torch.tensor(self.labels)
		self.labels = (self.labels).long()


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):

		sensor_data1 = self.data1[idx]
		sensor_data1 = torch.unsqueeze(sensor_data1, 0)

		sensor_data2 = self.data2[idx]
		sensor_data2 = torch.unsqueeze(sensor_data2, 0)

		sensor_data3 = self.data3[idx]
		sensor_data3 = torch.unsqueeze(sensor_data3, 0)

		activity_label = self.labels[idx]

		return sensor_data1, sensor_data2, sensor_data3, activity_label


class Unimodal_dataset():
	"""Build dataset from motion sensor data."""
	def __init__(self, x, y):

		self.data = x.tolist() #concate and tolist
		self.labels = y.tolist() #tolist

		self.data = torch.tensor(self.data) # to tensor
		self.labels = torch.tensor(self.labels)
		self.labels = (self.labels).long()


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):

		sensor_data = self.data[idx]
		sensor_data = torch.unsqueeze(sensor_data, 0)

		activity_label = self.labels[idx]

		return sensor_data, activity_label

class Unimodal_dataset_idx():
	"""Build dataset from motion sensor data."""
	def __init__(self, x, y):

		self.data = x.tolist() #concate and tolist
		self.labels = y.tolist() #tolist

		self.data = torch.tensor(self.data) # to tensor
		self.labels = torch.tensor(self.labels)
		self.labels = (self.labels).long()


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):

		sensor_data = self.data[idx]
		sensor_data = torch.unsqueeze(sensor_data, 0)

		activity_label = self.labels[idx]

		return idx, sensor_data, activity_label
	

def sensor_data_normalize_all(sensor_str, data, dataset):

	if dataset == "Realworld":
		MEAN_OF_IMU = MEAN_OF_IMU_Realworld
		STD_OF_IMU = STD_OF_IMU_Realworld
	elif dataset == "Shoaib":
		MEAN_OF_IMU = MEAN_OF_IMU_Shoaib
		STD_OF_IMU = STD_OF_IMU_Shoaib
	elif dataset == "Motionsense":
		MEAN_OF_IMU = MEAN_OF_IMU_Motionsense
		STD_OF_IMU = STD_OF_IMU_Motionsense		

	if sensor_str == 'acc':
		data = (data - MEAN_OF_IMU[0]) / STD_OF_IMU[0]

	elif sensor_str == 'gyro':
		data = (data - MEAN_OF_IMU[1]) / STD_OF_IMU[1]

	elif sensor_str == 'mag':
		data = (data - MEAN_OF_IMU[2]) / STD_OF_IMU[2]

	return data


## load IMU data
def load_data_IMU(dataset):

	folder_path = all_data_folder + dataset

	x1 = []
	x2 = []
	x3 = []
	y = np.load(folder_path + "/label.npy")

	for sample_id in range(y.shape[0]):

		acc_data = np.load(folder_path + "/acc/{}.npy".format(sample_id))
		gyro_data = np.load(folder_path + "/gyro/{}.npy".format(sample_id))
		x1.append(acc_data)
		x2.append(gyro_data)
		
		if dataset != "Motionsense":
			mag_data = np.load(folder_path + "/mag/{}.npy".format(sample_id))
			x3.append(mag_data)

	x1 = np.array(x1)
	x2 = np.array(x2)
	x3 = np.array(x3)

	x1 = sensor_data_normalize_all('acc', x1, dataset)
	x2 = sensor_data_normalize_all('gyro', x2, dataset)
	if dataset != "Motionsense":
		x3 = sensor_data_normalize_all('mag', x3, dataset)

	print(x1.shape)
	print(x2.shape)
	print(x3.shape)
	print(y.shape)

	return x1, x2, x3, y


## load paired data by mmbind
class Multimodal_paired_dataset():
	"""Build dataset from motion sensor data."""
	def __init__(self, x1, x2, x3, similarity):

		self.data1 = x1.tolist() #concate and tolist
		self.data2 = x2.tolist() #concate and tolist
		self.data3 = x3.tolist() #concate and tolist
		# self.labels = y.tolist() #tolist
		self.similarity = similarity.tolist() #tolist

		self.data1 = torch.tensor(self.data1) # to tensor
		self.data2 = torch.tensor(self.data2) # to tensor
		self.data3 = torch.tensor(self.data3) # to tensor
		# self.labels = torch.tensor(self.labels)
		self.similarity = torch.tensor(self.similarity)
		# self.labels = (self.labels).long()


	def __len__(self):
		return len(self.similarity)

	def __getitem__(self, idx):

		sensor_data1 = self.data1[idx]
		sensor_data1 = torch.unsqueeze(sensor_data1, 0)

		sensor_data2 = self.data2[idx]
		sensor_data2 = torch.unsqueeze(sensor_data2, 0)

		sensor_data3 = self.data3[idx]
		sensor_data3 = torch.unsqueeze(sensor_data3, 0)

		# activity_label = self.labels[idx]
		activity_similarity = self.similarity[idx]

		return sensor_data1, sensor_data2, sensor_data3, activity_similarity


def load_label_paired_data(data_path):

	folder_path = "./label_paired_data_Motionsense_Shoaib/{}/".format(data_path)

	x1 = np.load(folder_path + "acc.npy")
	x2 = np.load(folder_path + "gyro.npy")
	x3 = np.load(folder_path + "mag.npy")
	y = np.load(folder_path + "label.npy")

	return x1, x2, x3, y


def load_paired_data(data_path):

	folder_path = "./save_mmbind/acc_paired_data_Motionsense_Shoaib/{}/".format(data_path)

	x1 = []
	x2 = []
	x3 = []
	# y = np.load(folder_path + "/label.npy")
	similarity = np.load(folder_path + "similarity.npy")

	for sample_id in range(similarity.shape[0]):

		x1.append(np.load(folder_path + "acc/{}.npy".format(sample_id)))
		x2.append(np.load(folder_path + "gyro/{}.npy".format(sample_id)))
		x3.append(np.load(folder_path + "mag/{}.npy".format(sample_id)))

	x1 = np.array(x1)
	x2 = np.array(x2)
	x3 = np.array(x3)
	
	print(x1.shape)
	print(x2.shape)
	print(x3.shape)
	print(similarity.shape)

	return x1, x2, x3, similarity


def load_all_paired_data():

	x1_A, x2_A, x3_A, y_A = load_paired_data("acc_remain_gyro_pair")
	x1_B, x2_B, x3_B, y_B = load_paired_data("acc_remain_mag_pair")

	x1 = np.vstack((x1_A, x1_B))
	x2 = np.vstack((x2_A, x2_B))
	x3 = np.vstack((x3_A, x3_B))
	y = np.hstack((y_A, y_B))

	print(x1.shape)
	print(x2.shape)
	print(x3.shape)
	print(y.shape)

	return x1, x2, x3, y


