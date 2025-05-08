import numpy as np
import torch
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import Dataset
import os
import pickle
from torchvision.transforms import v2
from PIL import Image
from tqdm import tqdm
import collections
class ResizeIfSmaller:
    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, img):
        # Check if width or height is smaller than the minimum size
        if img.width < self.min_size[0] or img.height < self.min_size[1]:
            img = v2.Resize(self.min_size)(img)
        return img    

valid_labels = {'bedroom':0, 'furniture_store':1, 'classroom':2, 'bathroom':3, 'rest_space':4}

# Train A will consist of Image + SemSeg
class TrainA(Dataset):
	def __init__(self, path = '/home/jason/Downloads/SUNRGBD_processed/', complete = False):
		valid_split = torch.load(path + 'trainA.pt')
		self.img = []
		self.semseg = []
		self.depth = [] # Get depth only if I pass in complete
		self.labels = []
		self.complete = complete

		with open(path + '/labels.txt') as handle:
			lines = [line.rstrip() for line in handle]
			label_dict = {item.split(',')[0] : item.split(',')[1] for item in lines}

		for folder in os.listdir(path):
			sensor = os.path.join(path, folder)
			if os.path.isdir(sensor):
				for file in os.listdir(sensor + '/img'):
					if int(file.split('.')[0]) in valid_split:
						self.img.append(sensor + '/img/' + file)
						self.depth.append(sensor + '/depth/' + file.split('.')[0] + '.png')
						self.semseg.append(sensor + '/semseg/' + file)
						self.labels.append(valid_labels[label_dict[file.split('.')[0]]])
		self.transform = v2.Compose([
				ResizeIfSmaller((480, 640)),
                v2.CenterCrop((480, 640)),
                v2.Resize((240, 320)),
                v2.ToTensor()
            ])
		
	def __len__(self):
		return len(self.img)

	def __getitem__(self, idx):
		if self.complete:
			return {
				'img': self.transform(Image.open(self.img[idx])),
				'semseg': self.transform(Image.open(self.semseg[idx])),
				'depth': self.transform(Image.open(self.depth[idx])) / 2 ** 16,
				'label': self.labels[idx],
				'mask': torch.tensor([1, 1, 0]),
				'similarity': torch.tensor([1.0])
			}
		return {
			'img': self.transform(Image.open(self.img[idx])),
			'semseg': self.transform(Image.open(self.semseg[idx])),
			'depth': torch.zeros((1, 240, 320)),
			'label': self.labels[idx],
			'mask': torch.tensor([1, 1, 0]),
			'similarity': torch.tensor([1.0])
		}

# Train B will have Image + Depth
class TrainB(Dataset):
	def __init__(self, path = '/home/jason/Downloads/SUNRGBD_processed/', complete = False):
		self.img = []
		self.semseg = [] #  Get semseg only if I pass in complete
		self.depth = []
		self.labels = []
		self.complete = complete

		with open(path + '/labels.txt') as handle:
			lines = [line.rstrip() for line in handle]
			label_dict = {item.split(',')[0] : item.split(',')[1] for item in lines}

		valid_split = torch.load(path + 'trainB.pt')
		for folder in os.listdir(path):
			sensor = os.path.join(path, folder)
			if os.path.isdir(sensor):
				for file in os.listdir(sensor + '/img'):
					if int(file.split('.')[0]) in valid_split:
						self.img.append(sensor + '/img/' + file)
						self.depth.append(sensor + '/depth/' + file.split('.')[0] + '.png')
						self.semseg.append(sensor + '/semseg/' + file)
						self.labels.append(valid_labels[label_dict[file.split('.')[0]]])
		self.transform = v2.Compose([
				ResizeIfSmaller((480, 640)),
                v2.CenterCrop((480, 640)),
                v2.Resize((240, 320)),
                v2.ToTensor()
            ])
	def __len__(self):
		return len(self.img)

	def __getitem__(self, idx):
		if self.complete:
			return {
				'img': self.transform(Image.open(self.img[idx])),
				'semseg': self.transform(Image.open(self.semseg[idx])),
				'depth': self.transform(Image.open(self.depth[idx])) / 2 ** 16,
				'label': self.labels[idx],
				'mask': torch.tensor([1, 0, 1]),
				'similarity': torch.tensor([1.0])
			}
		return {
			'img': self.transform(Image.open(self.img[idx])),
			'semseg': torch.zeros((3, 240, 320)),
			'depth': self.transform(Image.open(self.depth[idx])) / 2 ** 16,
			'label': self.labels[idx],
			'mask': torch.tensor([1, 0, 1]),
			'similarity': torch.tensor([1.0])
		}


class FinetuneDataset(Dataset):
	def __init__(self, path = '/home/jason/Downloads/SUNRGBD_processed/'):
		self.img = []
		self.semseg = [] #  Get semseg only if I pass in complete
		self.depth = []
		self.labels = []

		with open(path + '/labels.txt') as handle:
			lines = [line.rstrip() for line in handle]
			label_dict = {item.split(',')[0] : item.split(',')[1] for item in lines}

		valid_split = torch.load(path + 'finetune.pt')
		valid_split = valid_split[:len(valid_split) // 2]
		for folder in os.listdir(path):
			sensor = os.path.join(path, folder)
			if os.path.isdir(sensor):
				for file in os.listdir(sensor + '/img'):
					if int(file.split('.')[0]) in valid_split:
						self.img.append(sensor + '/img/' + file)
						self.depth.append(sensor + '/depth/' + file.split('.')[0] + '.png')
						self.semseg.append(sensor + '/semseg/' + file)
						self.labels.append(valid_labels[label_dict[file.split('.')[0]]])
		print("\nFinetune dataset breakdown", collections.Counter(self.labels))
		self.transform = v2.Compose([
				ResizeIfSmaller((480, 640)),
                v2.CenterCrop((480, 640)),
				v2.Resize((240, 320)),
				#v2.RandomAffine(degrees=20, translate=(0.1, 0.1)),
                v2.ToTensor()
            ])
	def __len__(self):
		return len(self.img)

	def __getitem__(self, idx):
		return {
			'img': self.transform(Image.open(self.img[idx])),
			'semseg': self.transform(Image.open(self.semseg[idx])),
			'depth': self.transform(Image.open(self.depth[idx])) / 2 ** 16,
			'label': self.labels[idx],
			'mask': torch.tensor([1, 0, 1]),
			'similarity': torch.tensor([1.0])
		}

class TestDataset(Dataset):
	def __init__(self, path = '/home/jason/Downloads/SUNRGBD_processed/'):
		self.img = []
		self.semseg = [] #  Get semseg only if I pass in complete
		self.depth = []
		self.labels = []

		with open(path + '/labels.txt') as handle:
			lines = [line.rstrip() for line in handle]
			label_dict = {item.split(',')[0] : item.split(',')[1] for item in lines}

		valid_split = torch.load(path + 'test.pt')
		
		for folder in os.listdir(path):
			sensor = os.path.join(path, folder)
			if os.path.isdir(sensor):
				for file in os.listdir(sensor + '/img'):
					if int(file.split('.')[0]) in valid_split:
						self.img.append(sensor + '/img/' + file)
						self.depth.append(sensor + '/depth/' + file.split('.')[0] + '.png')
						self.semseg.append(sensor + '/semseg/' + file)
						self.labels.append(valid_labels[label_dict[file.split('.')[0]]])
		print("Test Dataset Breakdown", collections.Counter(self.labels))
		self.transform = v2.Compose([
				ResizeIfSmaller((480, 640)),
                v2.CenterCrop((480, 640)),
                v2.Resize((240, 320)),
                v2.ToTensor()
            ])
		
		self.img_data = []
		self.semseg_data = [] #  Get semseg only if I pass in complete
		self.depth_data = []

		for idx in tqdm(range(len(self.img))):
			self.img_data.append(self.transform(Image.open(self.img[idx])))
			self.semseg_data.append(self.transform(Image.open(self.semseg[idx])))
			self.depth_data.append(self.transform(Image.open(self.depth[idx])) / 2 ** 16)

	def __len__(self):
		return len(self.img)

	def __getitem__(self, idx):
		return {
			'img': self.img_data[idx],
			'semseg': self.semseg_data[idx],
			'depth': self.depth_data[idx],
			'label': self.labels[idx],
			'mask': torch.tensor([1, 0, 1]),
			'similarity': torch.tensor([1.0])
		}


class PickleDataset(Dataset):
	def __init__(self, root):
		self.data = []
		for file in os.listdir(root):
			if ('.pickle' in file):
				with open(root + '/' + file, 'rb') as handle:
					self.data.append(pickle.load(handle))
	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		return self.data[idx]
			

# def generate_dataset_splits(train_A_percent = 0.4, train_B_percent = 0.4, finetune_percent = 0.02):
# 	with open('/home/jason/Downloads/SUNRGBD_processed/labels.txt') as handle:
# 		lines = [line.rstrip() for line in handle]
# 		img_ids = torch.tensor([int(item.split(',')[0]) for item in lines if item.split(',')[1] in valid_labels.keys() ])
# 	num_elements = len(img_ids)
# 	noise = torch.rand(num_elements)
# 	selection = torch.argsort(noise)
# 	train_A = img_ids[selection[0: int(train_A_percent * num_elements)]]
# 	selection = selection[int(train_A_percent * num_elements): ]
# 	train_B = img_ids[selection[0: int(train_B_percent * num_elements)]]
# 	selection = selection[int(train_B_percent * num_elements): ]
# 	finetune = img_ids[selection[0: int(finetune_percent * num_elements)]]
# 	selection = selection[int(finetune_percent * num_elements): ]
# 	test = img_ids[selection]
# 	torch.save(train_A, '/home/jason/Downloads/SUNRGBD_processed/trainA.pt')
# 	torch.save(train_B, '/home/jason/Downloads/SUNRGBD_processed/trainB.pt')
# 	torch.save(finetune, '/home/jason/Downloads/SUNRGBD_processed/finetune.pt')
# 	torch.save(test, '/home/jason/Downloads/SUNRGBD_processed/test.pt')

# generate_dataset_splits()