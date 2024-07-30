import numpy as np
import torch
from sklearn.model_selection import train_test_split
import random

MEAN_OF_IMU = [-0.32627436907665514, -0.8661114601303396]
STD_OF_IMU = [0.6761486428324216, 113.55369543559192]
MEAN_OF_SKELETON = [-0.08385579666058844, -0.2913725901521685, 2.8711066708996738]
STD_OF_SKELETON = [0.14206656362043646, 0.4722835954035046, 0.16206781976658088]

random.seed(0)


all_data_folder = '../../UTD-split-222/'


## load original data
class Multimodal_dataset():
	"""Build dataset from motion sensor data."""
	def __init__(self, x1, x2, y):

		self.data1 = x1.tolist() #concate and tolist
		self.data2 = x2.tolist() #concate and tolist
		self.labels = y.tolist() #tolist

		self.data1 = torch.tensor(self.data1) # to tensor
		self.data2 = torch.tensor(self.data2) # to tensor
		self.labels = torch.tensor(self.labels)
		self.labels = (self.labels).long()


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):

		sensor_data1 = self.data1[idx]
		sensor_data1 = torch.unsqueeze(sensor_data1, 0)

		sensor_data2 = self.data2[idx]
		sensor_data2 = torch.unsqueeze(sensor_data2, 0)

		activity_label = self.labels[idx]

		return sensor_data1, sensor_data2, activity_label


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
	


def sensor_data_normalize_all(sensor_str, data):

	if sensor_str == 'acc':
		data = (data - MEAN_OF_IMU[0]) / STD_OF_IMU[0]

	elif sensor_str == 'gyro':
		data = (data - MEAN_OF_IMU[1]) / STD_OF_IMU[1]

	elif sensor_str == 'skeleton':
		for axis_id in range(3):
			# data = data
			data[:,:,:,axis_id] = (data[:,:,:,axis_id] - MEAN_OF_SKELETON[axis_id]) / STD_OF_SKELETON[axis_id]

	return data


## load skeleton data
def load_data_skeleton(data_path):

	folder_path = all_data_folder + data_path

	x1 = []
	y = np.load(folder_path + "/label.npy")

	for sample_id in range(y.shape[0]):
		x1.append(np.load(folder_path + "/skeleton/{}.npy".format(sample_id)))

	x1 = np.array(x1)
	y = np.array(y)

	x1 = x1.swapaxes(1,3).swapaxes(2,3)#(-1, 40, 20, 3)

	x1 = sensor_data_normalize_all('skeleton', x1)

	print(x1.shape)
	print(y.shape)

	return x1,y


def load_all_data_skeleton():

	x1_A, y_A = load_data_skeleton("train_A")
	x1_B, y_B = load_data_skeleton("train_B")

	x1 = np.vstack((x1_A, x1_B))
	y = np.hstack((y_A, y_B))

	print(x1.shape)
	print(y.shape)

	return x1, y



## load IMU data
def load_data_IMU(data_path):

	folder_path = all_data_folder + data_path

	x1 = []
	x2 = []
	y = np.load(folder_path + "/label.npy")

	for sample_id in range(y.shape[0]):

		inertial_data = np.load(folder_path + "/inertial/{}.npy".format(sample_id))
		x1.append(inertial_data[:, 0:3])
		x2.append(inertial_data[:, 3:6])

	x1 = np.array(x1)
	x2 = np.array(x2)

	x1 = sensor_data_normalize_all('acc', x1)
	x2 = sensor_data_normalize_all('gyro', x2)

	print(x1.shape)
	print(x2.shape)
	print(y.shape)

	return x1, x2, y


def load_all_data_IMU():

	x1_A, x2_A, y_A = load_data_IMU("train_A")
	x1_B, x2_B, y_B = load_data_IMU("train_B")

	x1 = np.vstack((x1_A, x1_B))
	x2 = np.vstack((x2_A, x2_B))
	y = np.hstack((y_A, y_B))

	print(x1.shape)
	print(x2.shape)
	print(y.shape)

	return x1, x2, y


def load_single_dataset(data_path):

	if data_path == "train_A":
		x1, y = load_data_skeleton(data_path)#skeleton
		return x1, y

	elif data_path == "train_B":
		_, x1, y = load_data_IMU(data_path)#gyro
		return x1, y

	else:# train_C and test
		x1, _ = load_data_skeleton(data_path)#skeleton
		_, x2, y = load_data_IMU(data_path)#gyro
		return x1, x2, y
	

def load_all_dataset():

	x1_A, _ = load_data_skeleton("train_A")#skeleton
	_, x2_A, y_A = load_data_IMU("train_A")#gyro
	

	x1_B, _ = load_data_skeleton("train_B")#skeleton
	_, x2_B, y_B = load_data_IMU("train_B")#gyro
	

	x1 = np.vstack((x1_A, x1_B))
	x2 = np.vstack((x2_A, x2_B))
	y = np.hstack((y_A, y_B))

	print(x1.shape)
	print(x2.shape)
	print(y.shape)

	return x1, x2, y



## load paired data by mmbind
class Multimodal_paired_dataset():
	"""Build dataset from motion sensor data."""
	def __init__(self, x1, x2, y, similarity):

		self.data1 = x1.tolist() #concate and tolist
		self.data2 = x2.tolist() #concate and tolist
		self.labels = y.tolist() #tolist
		self.similarity = similarity.tolist() #tolist

		self.data1 = torch.tensor(self.data1) # to tensor
		self.data2 = torch.tensor(self.data2) # to tensor
		self.labels = torch.tensor(self.labels)
		self.labels = (self.labels).long()
		self.similarity = torch.tensor(self.similarity)


	def __len__(self):
		return len(self.similarity)

	def __getitem__(self, idx):

		sensor_data1 = self.data1[idx]
		sensor_data1 = torch.unsqueeze(sensor_data1, 0)

		sensor_data2 = self.data2[idx]
		sensor_data2 = torch.unsqueeze(sensor_data2, 0)

		activity_label = self.labels[idx]
		activity_similarity = self.similarity[idx]

		return sensor_data1, sensor_data2, activity_label, activity_similarity



def load_paired_data(data_path):

	folder_path = "./save_mmbind/{}/".format(data_path)

	x1 = []
	x2 = []
	# y = np.load(folder_path + "/label.npy")
	similarity = np.load(folder_path + "similarity.npy")

	for sample_id in range(similarity.shape[0]):

		x1.append(np.load(folder_path + "skeleton/{}.npy".format(sample_id)))
		x2.append(np.load(folder_path + "gyro/{}.npy".format(sample_id)))

	x1 = np.array(x1)
	x2 = np.array(x2)

	x1 = x1.swapaxes(1,3).swapaxes(2,3)#(-1, 40, 20, 3)

	x1 = sensor_data_normalize_all('skeleton', x1)
	x2 = sensor_data_normalize_all('gyro', x2)
	
	print(x1.shape)
	print(x2.shape)
	print(similarity.shape)

	return x1, x2, similarity



def load_all_paired_data():

	x1_A, x2_A, y_A = load_paired_data("train_skeleton_paired_AB")
	x1_B, x2_B, y_B = load_paired_data("train_gyro_paired_AB")

	x1 = np.vstack((x1_A, x1_B))
	x2 = np.vstack((x2_A, x2_B))
	y = np.hstack((y_A, y_B))

	print(x1.shape)
	print(x2.shape)
	print(y.shape)

	return x1, x2, y



## load data for masked input
class Multimodal_incomplete_dataset():
	"""Build dataset from motion sensor data."""
	def __init__(self, x1, x2, y, mask):

		self.data1 = x1.tolist() #concate and tolist
		self.data2 = x2.tolist() #concate and tolist
		self.labels = y.tolist() #tolist
		self.mask = mask.tolist()

		self.data1 = torch.tensor(self.data1) # to tensor
		self.data2 = torch.tensor(self.data2) # to tensor
		self.labels = torch.tensor(self.labels)
		self.labels = (self.labels).long()
		self.mask = torch.tensor(self.mask)


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):

		sensor_data1 = self.data1[idx]
		sensor_data1 = torch.unsqueeze(sensor_data1, 0)

		sensor_data2 = self.data2[idx]
		sensor_data2 = torch.unsqueeze(sensor_data2, 0)

		activity_label = self.labels[idx]

		sensor_mask = self.mask[idx]
		sensor_mask = torch.unsqueeze(sensor_mask, 0)

		return sensor_data1, sensor_data2, activity_label, sensor_mask


def load_data_incomplete(data_path):

	folder_path = all_data_folder + data_path

	x1 = []
	x2 = []
	y = np.load(folder_path + "/label.npy")

	for sample_id in range(y.shape[0]):

		inertial_data = np.load(folder_path + "/inertial/{}.npy".format(sample_id))
		skeleton_data = np.load(folder_path + "/skeleton/{}.npy".format(sample_id))
		x1.append(skeleton_data)
		x2.append(inertial_data[:, 3:6])

	x1 = np.array(x1)
	x2 = np.array(x2)
	y = np.array(y)

	x1 = x1.swapaxes(1,3).swapaxes(2,3)#(-1, 40, 20, 3)

	x1 = sensor_data_normalize_all('skeleton', x1)
	x2 = sensor_data_normalize_all('gyro', x2)

	print(x1.shape)
	print(x2.shape)
	print(y.shape)

	mask_vector = np.zeros((y.shape[0], 2, 128))

	if data_path == "train_A":
		x2 = np.zeros_like(x2)# Skeleton
		# x3 = np.random.randn(*(x3.shape))
		mask_vector[:, 0, :] = 1.0
	elif data_path == "train_B":
		x1 = np.zeros_like(x1)# Gyro
		# x1 = np.random.randn(*(x1.shape))
		mask_vector[:, 1, :] = 1.0
	else:
		mask_vector[:, 0, :] = 1.0# Skeleton, Gyro
		mask_vector[:, 1, :] = 1.0

	return x1, x2, y, mask_vector


def load_all_data_incomplete():

	x1_A, x2_A, y_A, mask_A = load_data_incomplete("train_A")
	x1_B, x2_B, y_B, mask_B = load_data_incomplete("train_B")

	x1 = np.vstack((x1_A, x1_B))
	x2 = np.vstack((x2_A, x2_B))
	y = np.hstack((y_A, y_B))
	mask = np.vstack((mask_A, mask_B))

	print(x1.shape)
	print(x2.shape)
	print(y.shape)
	print(mask.shape)

	return x1, x2, y, mask


def load_data_incomplete_attach(data_path):

	folder_path = all_data_folder + data_path

	x1 = []
	x2 = []
	y = np.load(folder_path + "/label.npy")

	for sample_id in range(y.shape[0]):

		inertial_data = np.load(folder_path + "/inertial/{}.npy".format(sample_id))
		skeleton_data = np.load(folder_path + "/skeleton/{}.npy".format(sample_id))
		x1.append(skeleton_data)
		x2.append(inertial_data[:, 3:6])

	x1 = np.array(x1)
	x2 = np.array(x2)
	y = np.array(y)

	x1 = x1.swapaxes(1,3).swapaxes(2,3)#(-1, 40, 20, 3)

	x1 = sensor_data_normalize_all('skeleton', x1)
	x2 = sensor_data_normalize_all('gyro', x2)

	print(x1.shape)
	print(x2.shape)
	print(y.shape)

	mask_vector = np.zeros((y.shape[0], 2))

	if data_path == "train_A":
		x2 = np.zeros_like(x2)#skeleton
		# x3 = np.random.randn(*(x3.shape))
		mask_vector[:, 0] = 1.0
	elif data_path == "train_B":
		x1 = np.zeros_like(x1)#gyro
		# x1 = np.random.randn(*(x1.shape))
		mask_vector[:, 1] = 1.0
	else:
		# x2 = np.random.randn(*(x2.shape))
		mask_vector[:, 0] = 1.0#skeleton, gyro
		mask_vector[:, 1] = 1.0

	return x1, x2, y, mask_vector


def load_all_data_incomplete_attach():

	x1_A, x2_A, y_A, mask_A = load_data_incomplete_attach("train_A")
	x1_B, x2_B, y_B, mask_B = load_data_incomplete_attach("train_B")

	x1 = np.vstack((x1_A, x1_B))
	x2 = np.vstack((x2_A, x2_B))
	y = np.hstack((y_A, y_B))
	mask = np.vstack((mask_A, mask_B))

	print(x1.shape)
	print(x2.shape)
	print(y.shape)
	print(mask.shape)

	return x1, x2, y, mask

