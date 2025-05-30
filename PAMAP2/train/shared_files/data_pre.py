import numpy as np
import torch
from sklearn.model_selection import train_test_split
import random
import os
from modules.print_utils import pprint

MEAN_OF_ACC = [-5.352440841728246, 3.032396945527465, 3.512015293987129]
STD_OF_ACC = [5.965133282949386, 3.895971002273014, 3.5137488686049982]
MEAN_OF_GYRO = [-0.016070265747710952, 0.013177160488001459, -0.014856555083432988]
STD_OF_GYRO = [1.302615298447328, 0.8196249263104076, 1.3700164141603475]
MEAN_OF_MAG = [22.222441311016556, -12.091423635802593, -24.273813063531204]
STD_OF_MAG = [24.342509398861992, 24.250325849759587, 21.946251902327646]

random.seed(0)


all_data_folder = '../../UTD-split-0507/UTD-split-0507-222-1/'

label_mapping = np.array([-1, 1, -1, 2, 3, -1, -1, -1, -1, -1, -1, -1, 4, 5, -1, -1, 6, 7])
## load original data
class Multimodal_dataset():
	"""Build dataset from motion sensor data."""
	def __init__(self, valid_actions, valid_mods, root ='train_C', opt=None, data_duration=1000):
		self.data_arr = []
		self.labels = []
		self.valid_mods = valid_mods
		self.file_names = []

		if "train_all_paired_AB" in root:
			index_file_path = root
			index_files = os.listdir(root)
		elif "generated_AB" in root:
			index_file_path = root
			index_files = os.listdir(root)
		else:
			index_file_path = os.path.join(opt.indice_file, f"{root}.txt")
			index_files = np.loadtxt(index_file_path, dtype=str)

		if "train_all_paired_AB" in root:
			self.similarity_A = np.load(root + "similarity_setA.npy", allow_pickle=True).item()
			self.similarity_B = np.load(root + "similarity_setB.npy", allow_pickle=True).item()
			self.similarity = []
		else:
			self.similarity = None

		pprint(f"Loading Multimodal {valid_mods} datasets from {opt.processed_data_path} - {index_file_path}")
		print(f"=\tLoading Multimodal {valid_mods} datasets from {opt.processed_data_path} - {index_file_path}")

		for file in sorted(index_files):
			if file[-4:] != ".npy" or "generated_AB" in root or "similarity" in file:
				continue

			file_name = root + file if "train_all_paired_AB" in root else os.path.join(opt.processed_data_path, file)
			self.data_arr.append(np.load(file_name, allow_pickle=True))
			
			last_data = self.data_arr[-1]

			if "train_all_paired_AB" in root or "generated_AB" in root:
				label = int(file.split('_')[0])
				set_id = file.split('_')[1]
				file_counter = file.split('_')[2].split('.')[0]
				self.labels.append(label)

				if set_id == "setA":
					self.similarity.append(self.similarity_A[f'{label}_{set_id}_{file_counter}'])
				elif set_id == "setB":
					self.similarity.append(self.similarity_B[f'{label}_{set_id}_{file_counter}'])
				else:
					raise Exception(f'{label}_{set_id}_{file_counter}')
			else:
				self.labels.append(int(file.split('_')[1]))
			


			self.file_names.append(os.path.abspath(file_name))
		
		self.data_arr = np.array(self.data_arr)
		self.labels = np.array(self.labels)
	
	def __len__(self):
		return len(self.labels)

	# Currently this is wrist only, can expand the dictionary later to include more
	def __getitem__(self, idx):
		data = {}
		if ('acc' in self.valid_mods):
			data['acc'] = (self.data_arr[idx][:, 4:6 + 1] - MEAN_OF_ACC) / STD_OF_ACC
			data['acc'] = np.float32(data['acc'])
		if ('gyro' in self.valid_mods):
			data['gyro'] = (self.data_arr[idx][:, 10:12 + 1] - MEAN_OF_GYRO) / STD_OF_GYRO
			data['gyro'] =  np.float32(data['gyro'])
		if ('mag' in self.valid_mods):
			data['mag'] = (self.data_arr[idx][:, 13:15 + 1] - MEAN_OF_MAG) / STD_OF_MAG
			data['mag'] =  np.float32(data['mag'])
		data['action'] = label_mapping[self.labels[idx]] - 1 # zero index
		data['valid_mods'] = self.valid_mods
		data['data_path'] = self.file_names[idx]
		if self.similarity is not None:
			data['similarity'] = self.similarity[idx]
		return data

class Unimodal_dataset_direct_load():
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
		activity_label = self.labels[idx]

		return sensor_data, activity_label

class Multimodal_dataset_direct_load():
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
		# sensor_data1 = torch.unsqueeze(sensor_data1, 0)

		sensor_data2 = self.data2[idx]
		# sensor_data2 = torch.unsqueeze(sensor_data2, 0)

		activity_label = self.labels[idx]

		return (sensor_data1, sensor_data2), activity_label

## load data for masked input
class Multimodal_incomplete_dataset_direct_load():
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
		# sensor_data1 = torch.unsqueeze(sensor_data1, 0)

		sensor_data2 = self.data2[idx]
		# sensor_data2 = torch.unsqueeze(sensor_data2, 0)

		activity_label = self.labels[idx]

		sensor_mask = self.mask[idx]
		# sensor_mask = torch.unsqueeze(sensor_mask, 0)

		return sensor_data1, sensor_data2, activity_label, sensor_mask
		

class Multimodal_generated_dataset():
	"""Build dataset from motion sensor data."""
	def __init__(self, valid_actions, valid_mods, root ='train_C', opt=None, data_duration=1000):
		self.data_arr = []
		self.labels = []
		self.valid_mods = valid_mods
		self.file_names = []

		index_file_path = root
		index_files = os.listdir(root)

		pprint(f"Loading generated Multimodal {valid_mods} datasets from {index_file_path}")
		print(f"=\tLoading generated Multimodal {valid_mods} datasets from {index_file_path}")

		data_arr = []

		mod1, mod2 = opt.valid_mod[0][1], opt.valid_mod[1][1]
		for mod in [mod1, mod2]:
			mod_data_arr = []
			for file in sorted(os.listdir(os.path.join(root, mod))):
				if file[-4:] != ".npy":
					continue

				file_name = os.path.join(root, mod, file)
				mod_data = np.load(file_name, allow_pickle=True)
				mod_data_arr.append(mod_data)
			mod_data_arr = np.array(mod_data_arr)
			data_arr.append(mod_data_arr)


		self.data_arr = np.stack(data_arr)
		self.labels = np.load(os.path.join(root, "label.npy"))
	
	def __len__(self):
		return len(self.labels)

	# Currently this is wrist only, can expand the dictionary later to include more
	def __getitem__(self, idx):
		data = {}
		data[self.valid_mods[0]] = self.data_arr[0][idx]
		data[self.valid_mods[1]] = self.data_arr[1][idx]

		data['action'] = label_mapping[self.labels[idx]] - 1 # zero index
		data['valid_mods'] = self.valid_mods
		# data['data_path'] = self.file_names[idx]
		return data

class Multimodal_masked_dataset():
	"""Build dataset from motion sensor data."""
	def __init__(self, valid_actions, valid_mods, root ='train_C', opt=None, data_duration=1000):
		self.data_arr = []
		self.labels = []
		self.valid_mods = valid_mods
		self.file_names = []

		if "train_all_paired_AB" in root:
			index_file_path = root
			index_files = os.listdir(root)
		elif "generated_AB" in root:
			index_file_path = root
			index_files = os.listdir(root)
		else:
			index_file_path = os.path.join(opt.indice_file, f"{root}.txt")
			index_files = np.loadtxt(index_file_path, dtype=str)

		pprint(f"Loading Multimodal {valid_mods} datasets from {index_file_path}")
		print(f"=\tLoading Multimodal {valid_mods} datasets from {index_file_path}")

		for file in sorted(index_files):
			if file[-4:] != ".npy":
				continue

			file_name = os.path.join(opt.processed_data_path, file)
			self.data_arr.append(np.load(file_name, allow_pickle=True))
			self.labels.append(int(file.split('_')[1]))
			self.file_names.append(os.path.abspath(file_name))

		self.data_arr = np.array(self.data_arr)
		self.labels = np.array(self.labels)
		print(f"=\t{root} dataset shape: {self.data_arr.shape} with valid mod {self.valid_mods}")

		self.generate_mask()
	
	def generate_mask(self):
		print(f"=\tGenerating Mask vector")
		mask_vector = np.zeros((self.labels.shape[0], 3))
		mod_index = dict(zip(["acc", "gyro", "mag"], [0, 1, 2]))
		mask_vector[mod_index[self.valid_mods[0]]] = 1.0 # set common modality as 1
		if len(self.valid_mods) > 1:
			mask_vector[mod_index[self.valid_mods[1]]] = 1.0 # set other modality as 0
		self.masks = mask_vector
	
	def __len__(self):
		return len(self.labels)

	# Currently this is wrist only, can expand the dictionary later to include more
	def __getitem__(self, idx):
		data = {}

		mask = [1.0, 1.0, 1.0]

		data['acc'] = (self.data_arr[idx][:, 4:6 + 1] - MEAN_OF_ACC) / STD_OF_ACC
		data['acc'] = np.float32(data['acc'])

		data['gyro'] = (self.data_arr[idx][:, 10:12 + 1] - MEAN_OF_GYRO) / STD_OF_GYRO
		data['gyro'] =  np.float32(data['gyro'])

		data['mag'] = (self.data_arr[idx][:, 13:15 + 1] - MEAN_OF_MAG) / STD_OF_MAG
		data['mag'] =  np.float32(data['mag'])

		for i, mod in enumerate(["acc", "gyro", "mag"]):
			if mod not in self.valid_mods:
				data[mod] = np.zeros_like(data[mod])
				mask[i] = 0.0

		data['action'] = label_mapping[self.labels[idx]] - 1 # zero index
		data['valid_mods'] = self.valid_mods
		data['data_path'] = self.file_names[idx]
		data['mask'] = np.float32(np.array(mask))

		return data


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

	x1 = sensor_data_normalize_all('skeleton', x2)

	print(x1.shape)
	print(y.shape)

	return x1,y


def load_all_data_skeleton():

	x1_A, y_A = load_data("train_A")
	x1_B, y_B = load_data("train_B")

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

	x1_A, x2_A, y_A = load_data("train_A")
	x1_B, x2_B, y_B = load_data("train_B")

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
	def __init__(self, x1, x2, similarity):

		self.data1 = x1.tolist() #concate and tolist
		self.data2 = x2.tolist() #concate and tolist
		# self.labels = y.tolist() #tolist
		self.similarity = similarity.tolist() #tolist

		self.data1 = torch.tensor(self.data1) # to tensor
		self.data2 = torch.tensor(self.data2) # to tensor
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

		# activity_label = self.labels[idx]
		activity_similarity = self.similarity[idx]

		return sensor_data1, sensor_data2, activity_similarity



def load_paired_data(data_path):

	folder_path = "./{}/".format(data_path)

	x1 = []
	x2 = []
	# y = np.load(folder_path + "/label.npy")
	similarity = np.load(folder_path + "similarity.npy")

	for sample_id in range(similarity.shape[0]):

		x1.append(np.load(folder_path + "acc/{}.npy".format(sample_id)))
		x2.append(np.load(folder_path + "gyro/{}.npy".format(sample_id)))

	x1 = np.array(x1)
	x2 = np.array(x2)

	x1 = sensor_data_normalize_all('acc', x1)
	x2 = sensor_data_normalize_all('gyro', x2)
	
	print(x1.shape)
	print(x2.shape)
	print(similarity.shape)

	return x1, x2, similarity



def load_all_paired_data():

	x1_A, x2_A, y_A = load_paired_data("train_acc_paired_AB")
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
	def __init__(self, x1, x2, x3, y, mask):

		self.data1 = x1.tolist() #concate and tolist
		self.data2 = x2.tolist() #concate and tolist
		self.data3 = x3.tolist()
		self.labels = y.tolist() #tolist
		self.mask = mask.tolist()

		self.data1 = torch.tensor(self.data1) # to tensor
		self.data2 = torch.tensor(self.data2) # to tensor
		self.data3 = torch.tensor(self.data3)
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

		sensor_data3 = self.data3[idx]
		sensor_data3 = torch.unsqueeze(sensor_data3, 0)

		activity_label = self.labels[idx]

		sensor_mask = self.mask[idx]
		sensor_mask = torch.unsqueeze(sensor_mask, 0)

		return sensor_data1, sensor_data2, sensor_data3, activity_label, sensor_mask


def load_data_incomplete(data_path):

	folder_path = all_data_folder + data_path

	x1 = []
	x2 = []
	x3 = []
	y = np.load(folder_path + "/label.npy")

	for sample_id in range(y.shape[0]):

		inertial_data = np.load(folder_path + "/inertial/{}.npy".format(sample_id))
		skeleton_data = np.load(folder_path + "/skeleton/{}.npy".format(sample_id))
		x1.append(inertial_data[:, 0:3])
		x2.append(skeleton_data)
		x3.append(inertial_data[:, 3:6])

	x1 = np.array(x1)
	x2 = np.array(x2)
	x3 = np.array(x3)
	y = np.array(y)

	x2 = x2.swapaxes(1,3).swapaxes(2,3)#(-1, 40, 20, 3)

	x1 = sensor_data_normalize_all('acc', x1)
	x2 = sensor_data_normalize_all('skeleton', x2)
	x3 = sensor_data_normalize_all('gyro', x3)

	print(x1.shape)
	print(x2.shape)
	print(x3.shape)
	print(y.shape)

	mask_vector = np.zeros((y.shape[0], 3, 128))

	if data_path == "train_A":
		x3 = np.zeros_like(x3)
		mask_vector[:, 0, :] = 1.0
		mask_vector[:, 1, :] = 1.0
	elif data_path == "train_B":
		x1 = np.zeros_like(x1)
		mask_vector[:, 1, :] = 1.0
		mask_vector[:, 2, :] = 1.0
	else:
		x2 = np.zeros_like(x2)
		mask_vector[:, 0, :] = 1.0
		mask_vector[:, 2, :] = 1.0

	return x1, x2, x3, y, mask_vector



def load_all_data_incomplete():

	x1_A, x2_A, x3_A, y_A, mask_A = load_data("train_A")
	x1_B, x2_B, x3_B, y_B, mask_B = load_data("train_B")

	x1 = np.vstack((x1_A, x1_B))
	x2 = np.vstack((x2_A, x2_B))
	x3 = np.vstack((x3_A, x3_B))
	y = np.hstack((y_A, y_B))
	mask = np.vstack((mask_A, mask_B))

	print(x1.shape)
	print(x2.shape)
	print(x3.shape)
	print(y.shape)
	print(mask.shape)

	return x1, x2, x3, y, mask


if __name__ == '__main__':
	test = Multimodal_dataset([1, 3, 4, 12, 13, 16, 17], ['acc', 'gyro', 'mag'])

