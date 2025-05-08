import torch
from torch.utils.data import Dataset
import os 
from PIL import Image
from torchvision.transforms import v2

class KinectDataset(Dataset):
    def __init__(self, path = '/home/jason/Downloads/SUNRGBD/kv2/kinect2data/', transform = None):
        self.file_names = []
        self.labels = []
        for dir in os.listdir(path):
            img_folder = path + dir + '/image/'
            with open(path+dir + '/scene.txt' , 'r') as file:
                label = file.read()
            if label not in ['bathroom', 'office', 'classroom']:
                continue
            self.labels.append(label)
            for file in os.listdir(img_folder):
                self.file_names.append((img_folder + file))
        if transform:
            self.transform = transform
        else:
            self.transform = v2.Compose([
                v2.CenterCrop((480, 640)),
                v2.Resize((240, 320)),
                v2.ToTensor()
            ])
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        img = Image.open(self.file_names[index])
        return {
            'img': self.transform(img), 
            'label': self.labels[index]
        }

class ResizeIfSmaller:
    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, img):
        # Check if width or height is smaller than the minimum size
        if img.width < self.min_size[0] or img.height < self.min_size[1]:
            img = v2.Resize(self.min_size)(img)
        return img    

class SUNRGBD(Dataset):
    def __init__(self, path = '/home/jason/Downloads/SUNRGBD/', transform=None):
        self.data = []
        if transform:
            self.transform = transform
        else:
            self.transform = v2.Compose([
                ResizeIfSmaller((480, 640)),
                v2.CenterCrop((480, 640)),
                v2.Resize((240, 320)),
                v2.ToTensor()
            ])
        
        for subdir, _, files in os.walk(path):
            # Look for the image folder and scene.txt file in each subdirectory
            image_folder = os.path.join(subdir, 'image')
            scene_file = os.path.join(subdir, 'scene.txt')
            
            # Check if both the image folder and scene file exist
            if os.path.isdir(image_folder) and os.path.isfile(scene_file):
                # Find image file (assumes there's only one image)
                for img_file in os.listdir(image_folder):
                    if img_file.endswith(('.jpg', '.png')):
                        image_path = os.path.join(image_folder, img_file)
                        with open(scene_file, 'r') as f:
                            label = f.read().strip()
                        #if label in ['rest_space',  'bathroom', 'classroom', 'office', 'furniture_store', 'bedroom', 'living_room', 'kitchen']:
                            self.data.append((image_path, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        curr_img, curr_label = self.data[index]
        img = Image.open(curr_img)
        return {'img': self.transform(img), 
                'label': curr_label}
