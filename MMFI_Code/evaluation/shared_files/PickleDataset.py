import os
import scipy.io as scio
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle

# Mean and STD of RGB and mmWave to ensure no weird feature magnitudes
MEAN_MMWAVE = torch.tensor([ 3.0253e+00, -2.4614e-02, -7.2202e-02,  1.6478e+01, -1.8120e-03])
STD_MMWAVE = torch.tensor([0.6514, 0.6469, 1.0867, 3.9157, 0.7542])
MEAN_RGB = np.array([337.29, 243.4672])
STD_RGB = np.array([54.781, 101.1314])

# Function provided by MMFI to decode the yaml files
def decode_config(config):
    all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14',
                    'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28',
                    'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
    all_actions = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
                   'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27']
    train_form = {}
    val_form = {}
    # Limitation to actions (protocol)
    if config['protocol'] == 'protocol1':  # Daily actions
        actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
    elif config['protocol'] == 'protocol2':  # Rehabilitation actions:
        actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A15', 'A16', 'A24', 'A25', 'A26']
    else:
        actions = all_actions
    # Limitation to subjects and actions (split choices)
    if config['split_to_use'] == 'random_split':
        rs = config['random_split']['random_seed']
        ratio = config['random_split']['ratio']
        for action in actions:
            np.random.seed(rs)
            idx = np.random.permutation(len(all_subjects))
            idx_train = idx[:int(np.floor(ratio*len(all_subjects)))]
            idx_val = idx[int(np.floor(ratio*len(all_subjects))):]
            subjects_train = np.array(all_subjects)[idx_train].tolist()
            subjects_val = np.array(all_subjects)[idx_val].tolist()
            for subject in all_subjects:
                if subject in subjects_train:
                    if subject in train_form:
                        train_form[subject].append(action)
                    else:
                        train_form[subject] = [action]
                if subject in subjects_val:
                    if subject in val_form:
                        val_form[subject].append(action)
                    else:
                        val_form[subject] = [action]
            rs += 1
    elif config['split_to_use'] == 'cross_scene_split':
        subjects_train = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                          'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
                          'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']
        subjects_val = ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    elif config['split_to_use'] == 'cross_subject_split':
        subjects_train = config['cross_subject_split']['train_dataset']['subjects']
        subjects_val = config['cross_subject_split']['val_dataset']['subjects']
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    else:
        subjects_train = config['manual_split']['train_dataset']['subjects']
        subjects_val = config['manual_split']['val_dataset']['subjects']
        actions_train = config['manual_split']['train_dataset']['actions']
        actions_val = config['manual_split']['val_dataset']['actions']
        for subject in subjects_train:
            train_form[subject] = actions_train
        for subject in subjects_val:
            val_form[subject] = actions_val

    dataset_config = {'train_dataset': {'modality': config['modality'],
                                        'split': 'training',
                                        'data_form': train_form
                                        },
                      'val_dataset': {'modality': config['modality'],
                                      'split': 'validation',
                                      'data_form': val_form}}
    return dataset_config

# Class provided by MMFI where we creates dicts with the file paths of each sample
class MMFi_Database:
    def __init__(self, data_root):
        self.data_root = data_root
        self.scenes = {}
        self.subjects = {}
        self.actions = {}
        self.modalities = {}
        self.load_database()

    def load_database(self):
        for scene in sorted(os.listdir(self.data_root)):
            if scene.startswith("."):
                continue
            self.scenes[scene] = {}
            for subject in sorted(os.listdir(os.path.join(self.data_root, scene))):
                if subject.startswith("."):
                    continue
                self.scenes[scene][subject] = {}
                self.subjects[subject] = {}
                for action in sorted(os.listdir(os.path.join(self.data_root, scene, subject))):
                    if action.startswith("."):
                        continue
                    self.scenes[scene][subject][action] = {}
                    self.subjects[subject][action] = {}
                    if action not in self.actions.keys():
                        self.actions[action] = {}
                    if scene not in self.actions[action].keys():
                        self.actions[action][scene] = {}
                    if subject not in self.actions[action][scene].keys():
                        self.actions[action][scene][subject] = {}
                    for modality in ['infra1', 'infra2', 'depth', 'rgb', 'lidar', 'mmwave', 'wifi-csi']:
                        data_path = os.path.join(self.data_root, scene, subject, action, modality)
                        self.scenes[scene][subject][action][modality] = data_path
                        self.subjects[subject][action][modality] = data_path
                        self.actions[action][scene][subject][modality] = data_path
                        if modality not in self.modalities.keys():
                            self.modalities[modality] = {}
                        if scene not in self.modalities[modality].keys():
                            self.modalities[modality][scene] = {}
                        if subject not in self.modalities[modality][scene].keys():
                            self.modalities[modality][scene][subject] = {}
                        if action not in self.modalities[modality][scene][subject].keys():
                            self.modalities[modality][scene][subject][action] = data_path

# Custom defined dataset that reads from pickle files
class PickleDataset(Dataset):
    def __init__(self, data_base, data_unit, modality, split, data_form):
        self.data_base = data_base
        self.data_unit = data_unit
        self.modality = modality.split('|')
        for m in self.modality:
            assert m in ['rgb', 'infra1', 'infra2', 'depth', 'lidar', 'mmwave', 'wifi-csi']
        self.split = split
        self.data_source = data_form
        self.data_list = self.load_data()


    def get_scene(self, subject):
        if subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
            return 'E01'
        elif subject in ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']:
            return 'E02'
        elif subject in ['S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']:
            return 'E03'
        elif subject in ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']:
            return 'E04'
        else:
            raise ValueError('Subject does not exist in this dataset.')

    def get_data_type(self, mod):
        if mod in ["rgb", 'infra1', "infra2"]:
            return ".npy"
        elif mod in ["lidar", "mmwave"]:
            return ".bin"
        elif mod in ["depth"]:
            return ".png"
        elif mod in ["wifi-csi"]:
            return ".mat"
        else:
            raise ValueError("Unsupported modality.")

    # Called in the init function to grab the file path of the data
    def load_data(self):
        data_info = []
        for subject, actions in self.data_source.items():
            for action in actions:
                if self.data_unit == 'sequence':
                    data_dict = {'modality': self.modality,
                                 'scene': self.get_scene(subject),
                                 'subject': subject,
                                 'action': action,
                                 'gt_path': os.path.join(self.data_base.data_root, self.get_scene(subject), subject,
                                                         action, 'ground_truth.npy')
                                 }
                    for mod in self.modality:
                        data_dict[mod+'_path'] = os.path.join(self.data_base.data_root, self.get_scene(subject), subject,
                                                         action, mod)
                    data_info.append(data_dict)
                elif self.data_unit == 'frame':
                    frame_num = 297
                    for idx in range(frame_num):
                        data_dict = {'modality': self.modality,
                                     'scene': self.get_scene(subject),
                                     'subject': subject,
                                     'action': action,
                                     'gt_path': os.path.join(self.data_base.data_root, self.get_scene(subject), subject,
                                                             action, 'ground_truth.npy'),
                                     'idx': idx
                                     }
                        data_valid = True
                        for mod in self.modality:
                            data_dict[mod+'_path'] = os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, mod, "frame{:03d}".format(idx+1) + self.get_data_type(mod))

                            if os.path.getsize(data_dict[mod+'_path']) == 0:
                                data_valid = False
                        if data_valid:
                            data_info.append(data_dict)
                else:
                    raise ValueError('Unsupport data unit!')
        return data_info

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Get the appropriate file path at idx
        item = self.data_list[idx]
        chopIndex = item['gt_path'].rindex('/')
        data_path = item['gt_path'][:chopIndex]

        # Load pickle fle at the file path
        with open(data_path + '/data.pickle', 'rb') as handle:
            data = pickle.load(handle)

        key_arr = list(data.keys())
        key_arr = list(filter(lambda s: 'input' in s, key_arr))
        keys_exclude = list(filter(lambda s: s[6:] not in item['modality'], key_arr))
        # Get rid of data we don't care about
        for key in keys_exclude:
            data.pop(key)
        data['modality'] = item['modality']

        # Standardize the mean/var
        if 'input_mmwave' in data.keys():
            torch_data = [(torch.tensor(item).float() - MEAN_MMWAVE) / STD_MMWAVE for item in data['input_mmwave']]
            data['input_mmwave'] = nn.utils.rnn.pad_sequence(torch_data)
        if 'input_rgb' in data.keys():
            data['input_rgb'] -= MEAN_RGB
            data['input_rgb'] /= STD_RGB
        # For depth, we downsample from 240 x 320 to 48 x 64
        if 'input_depth' in data.keys():
            data['input_depth'] = np.reshape(data['input_depth'], (-1, 240, 320))
            data['input_depth'] = np.transpose(data['input_depth'], (1, 2, 0))
            data['input_depth'] = cv2.resize(data['input_depth'], (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
            
            data['input_depth'] = np.transpose(data['input_depth'], (2, 0, 1))
            data['input_depth'] = np.reshape(data['input_depth'], (-1, 3, 48, 64))
    

        return data
    

        gt_numpy = np.load(item['gt_path'])
        gt_torch = torch.from_numpy(gt_numpy)

        if self.data_unit == 'sequence':
            sample = {'modality': item['modality'],
                      'scene': item['scene'],
                      'subject': item['subject'],
                      'action': item['action'],
                      'output': gt_torch
                      }
            for mod in item['modality']:
                data_path = item[mod+'_path']
                sample['path'] = data_path
                data_path = data_path.replace('depth', 'depthColorized')
                if os.path.isdir(data_path):
                    data_mod = self.read_dir(data_path)
                else:
                    data_mod = np.load(data_path + '.npy')
                sample['input_'+mod] = data_mod
        elif self.data_unit == 'frame':
            sample = {'modality': item['modality'],
                      'scene': item['scene'],
                      'subject': item['subject'],
                      'action': item['action'],
                      'idx': item['idx'],
                      'output': gt_torch[item['idx']]
                      }
            for mod in item['modality']:
                data_path = item[mod + '_path']
                if os.path.isfile(data_path):
                    data_mod = self.read_frame(data_path)
                    sample['input_'+mod] = data_mod
                else:
                    raise ValueError('{} is not a file!'.format(data_path))
        else:
            raise ValueError('Unsupport data unit!')
        return sample

# make_datset is called by other files to createa a dataset, creates database and parses the yaml
# returns a train and validation dataset according to YAML
# However, I found that the validation dataset created is a bit weird, prefer to throw away the val_dataset returned
# and call make_dataset twice, i.e train_dataset, _  = make_dataset(...); val_dataset, _ = make_dataset(...)
def make_dataset(dataset_root, config):
    database = MMFi_Database(dataset_root)
    config_dataset = decode_config(config)
    train_dataset = PickleDataset(database, config['data_unit'], **config_dataset['train_dataset'])
    val_dataset = PickleDataset(database, config['data_unit'], **config_dataset['val_dataset'])
    return train_dataset, val_dataset


# def collate_fn_padd(batch):
#     '''
#     Padds batch of variable length
#     '''
#     import pdb; pdb.set_trace()
#     batch_data = {'modality': batch[0]['modality'],
#                   'scene': [sample['scene'] for sample in batch],
#                   'subject': [sample['subject'] for sample in batch],
#                   'action': [sample['action'] for sample in batch],
#                   'idx': [sample['idx'] for sample in batch] if 'idx' in batch[0] else None
#                   }
#     _output = [np.array(sample['output']) for sample in batch]
#     _output = torch.FloatTensor(np.array(_output))
#     batch_data['output'] = _output

#     for mod in batch_data['modality']:
#         if mod in ['mmwave', 'lidar']:
#             _input = [torch.Tensor(sample['input_' + mod]) for sample in batch]
#             _input = torch.nn.utils.rnn.pad_sequence(_input)
#             _input = _input.permute(1, 0, 2)
#             batch_data['input_' + mod] = _input
#         else:
#             _input = [np.array(sample['input_' + mod]) for sample in batch]
#             _input = torch.FloatTensor(np.array(_input))
#             batch_data['input_' + mod] = _input

#     return batch_data

# def make_dataloader(dataset, is_training, generator, batch_size, collate_fn_padd = collate_fn_padd):
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         collate_fn=collate_fn_padd,
#         shuffle=is_training,
#         drop_last=is_training,
#         generator=generator,
#         num_workers=10,
#         prefetch_factor=2

#     )
#     return loader


