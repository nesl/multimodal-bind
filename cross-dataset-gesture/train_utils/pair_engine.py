import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ConfusionMatrixDisplay

from data.gesture_dataset import normalize_dataset_mod
from train_utils.train_engine import val_loop
import utils.log as log


def plot_similarity_matrix(similarity_matrix, save_path, reference_dataset, search_dataset, filename="similarity_matrix"):
    plt.figure(figsize=(10, 10))
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    plt.title(f"Similarity Matrix between {reference_dataset} and {search_dataset}")
    plt.colorbar()
    plt.ylabel(f"{reference_dataset} samples")
    plt.xlabel(f"{search_dataset} samples")
    plt.savefig(os.path.join(save_path, f"{filename}.png"))
    plt.close()
    
    
def sensor_pair_engine(opt, model, loss_func, train_dataloader):
    
    # train_dataloader_1 === GR4DHCI (skeleton, stereo_ir)
    # train_dataloader_2 === DHG (skeleton, depth)
    train_dataloader_1, train_dataloader_2 = train_dataloader
    
    # Skeleton embedding from GR4DHCI and DHG
    log.divide("Extracting Skeleton Embedding from GR4DHCI")
    GR4DHCI_common_embedding, GR4DHCI_label, GR4DHCI_ids = val_loop(opt, None, model, loss_func, train_dataloader_1, None, return_features=True)
    
    log.divide("Extracting Skeleton Embedding from DHG")
    DHG_common_embedding, DHG_label, DHG_ids = val_loop(opt, None, model, loss_func, train_dataloader_2, None, return_features=True)
    
    opt.common_modality = "skeleton"
    if opt.dataset == "GR4DHCI":
        reference_feature = GR4DHCI_common_embedding
        reference_label = GR4DHCI_label
        reference_ids = GR4DHCI_ids
        search_feature = DHG_common_embedding
        search_label = DHG_label
        search_ids = DHG_ids
        opt.reference_modality = "stereo_ir"
        opt.search_modality = "depth"
        reference_dataset = "GR4DHCI"
        search_dataset = "DHG"
    elif opt.dataset == "DHG":
        reference_feature = DHG_common_embedding
        reference_label = DHG_label
        reference_ids = DHG_ids
        search_feature = GR4DHCI_common_embedding
        search_label = GR4DHCI_label
        search_ids = GR4DHCI_ids
        opt.reference_modality = "depth"
        opt.search_modality = "stereo_ir"
        reference_dataset = "DHG"
        search_dataset = "GR4DHCI"
    else:
        raise NotImplementedError

    save_paired_path = opt.save_path + f"paired_data_{opt.dataset}_{opt.reference_modality}/"
    if not os.path.exists(save_paired_path):
        os.makedirs(os.path.join(save_paired_path, 'skeleton'))
        os.makedirs(os.path.join(save_paired_path, 'stereo_ir'))
        os.makedirs(os.path.join(save_paired_path, 'depth'))
    
    log.logprint(f"Saving paired data at {save_paired_path}")
    reference_feature = torch.concat(reference_feature, dim=0)
    search_feature = torch.concat(search_feature, dim=0)
    reference_label = torch.concat(reference_label, dim=0)
    search_label = torch.concat(search_label, dim=0)
    reference_ids = torch.concat(reference_ids, dim=0)
    search_ids = torch.concat(search_ids, dim=0)
    
    
    similarity_matrix = cosine_similarity(reference_feature, search_feature)
    similarity_matrix_same = cosine_similarity(reference_feature, reference_feature)
    plot_similarity_matrix(similarity_matrix, save_paired_path, reference_dataset, search_dataset)
    plot_similarity_matrix(similarity_matrix_same, save_paired_path, reference_dataset, search_dataset, filename=f"{reference_dataset}_similarity_matrix")
    
    paired_data_length = reference_label.shape[0]
    select_label = np.zeros(paired_data_length)
    correct_map = 0
    
    similarity_record = np.zeros(paired_data_length)
    log.divide("Start pairing the data")
    for sample_index in tqdm(range(paired_data_length)):
        temp_similarity_vector = similarity_matrix[sample_index] # find the most similar one
        select_feature_index = np.argmax(temp_similarity_vector)
        select_label[sample_index] = search_label[select_feature_index]
        similarity_record[sample_index] = np.max(temp_similarity_vector)
        
        if reference_label[sample_index] == search_label[select_feature_index]:
            correct_map += 1

        # print(reference_label[sample_index], search_label[select_feature_index], select_feature_index, np.max(temp_similarity_vector), temp_correct)
        reference_sampel_id = reference_ids[sample_index]
        search_sample_id = search_ids[select_feature_index]
        
        common_data = np.load(os.path.join(opt.processed_data_path, reference_dataset, "skeleton", f"{reference_sampel_id}.npy"))
        reference_data = np.load(os.path.join(opt.processed_data_path, reference_dataset, opt.reference_modality, f"{reference_sampel_id}.npy"))
        search_data = np.load(os.path.join(opt.processed_data_path, search_dataset, opt.search_modality, f"{search_sample_id}.npy"))
        
        # normalize the dataset
        common_data = normalize_dataset_mod(common_data, opt.common_modality, reference_dataset)
        reference_data = normalize_dataset_mod(reference_data, opt.reference_modality, reference_dataset)
        search_data = normalize_dataset_mod(search_data, opt.search_modality, search_dataset)
        
        # Save the paired data
        np.save(os.path.join(save_paired_path, opt.common_modality, f"{sample_index}.npy"), common_data)
        np.save(os.path.join(save_paired_path, opt.reference_modality, f"{sample_index}.npy"), reference_data)
        np.save(os.path.join(save_paired_path, opt.search_modality, f"{sample_index}.npy"), search_data)

    np.save(save_paired_path + 'similarity.npy', similarity_record)
    np.save(save_paired_path + 'label.npy', reference_label)
    log.logprint(f"Correct map: {correct_map/paired_data_length}")

    disp = ConfusionMatrixDisplay.from_predictions(reference_label, select_label)
    disp.plot() 
    disp.ax_.set_title(f"Pair {reference_dataset} and {search_dataset} using {opt.common_modality} features")
    disp.ax_.set_ylabel('Label for {}'.format(reference_dataset))
    disp.ax_.set_xlabel('Selected Label from {}'.format(search_dataset))
    plt.savefig(save_paired_path + f"results_{opt.reference_modality}_pair")
    
    np.savetxt(save_paired_path + 'train.txt', np.arange(paired_data_length), fmt='%s')
    np.savetxt(save_paired_path + 'valid.txt', np.arange(paired_data_length), fmt='%s')

def label_pair_engine(opt, model, loss_func, train_dataloader):
    # train_dataloader_1 === GR4DHCI (skeleton, stereo_ir)
    # train_dataloader_2 === DHG (skeleton, depth)
    train_dataloader_1, train_dataloader_2 = train_dataloader
    def get_data_label(mod1, mod2, loader):
        mod_1_data = []
        mod_2_data = []
        all_labels = []
        log.logprint(f"Extracting {mod1} and {mod2} data")
        for data, labels, file_id in tqdm(loader):
            mod_1_data.append(data[mod1])
            mod_2_data.append(data[mod2])
            all_labels.append(labels)
        mod_1_data = torch.cat(mod_1_data, dim=0)
        mod_2_data = torch.cat(mod_2_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_labels = all_labels.numpy().flatten()
        print(mod_1_data.shape, mod_2_data.shape, all_labels.shape)
        return mod_1_data, mod_2_data, all_labels
    
    data_path = opt.processed_data_path.split("/")[-1]
    label_similarity_matrix = np.load(os.path.join(data_path, "label_similarity_matrix.npy")) # 8 * 14
    print(f"Loading label similarity matrix from {os.path.join(data_path, 'label_similarity_matrix.npy')}")
    log.logprint(f"Label Similarity Matrix shape: {label_similarity_matrix.shape}")

    skeleton_data_1, stereo_ir_data_1, labels_1 = get_data_label("skeleton", "stereo_ir", train_dataloader_1)
    skeleton_data_2, depth_data_2, labels_2 = get_data_label("skeleton", "depth", train_dataloader_2)
    
    log.logprint(f"GR4DHCI Skeleton shape: {skeleton_data_1[0].shape}")
    log.logprint(f"GR4DHCI labels: {labels_1.shape}")
    
    log.logprint(f"DHG Skeleton shape: {skeleton_data_2[0].shape}")
    log.logprint(f"DHG labels: {labels_2.shape}")
    
    labels_1 = [x for x in labels_1] # GR4DHCI labels start from 1 instead of zero
    labels_2 = [x - 1 for x in labels_2] # DHG labels start from 1 instead of zero
    labels_1 = np.array(labels_1)
    labels_2 = np.array(labels_2)
    
    # sanity check
    labels_set_1 = set(labels_1)
    num_class_1 = len(labels_set_1)
    labels_set_2 = set(labels_2)
    num_class_2 = len(labels_set_2)
    
    log.logprint(f"Start Pairing the data from GR4DHCI {label_similarity_matrix.shape[0]} classes and DHG {label_similarity_matrix.shape[1]} classes")
    
    print(min(labels_1), max(labels_1), min(labels_2), max(labels_2))
    log.logprint(f"Number of classes in GR4DHCI: {num_class_1}")
    log.logprint(f"Number of classes in DHG: {num_class_2}")
    
    index_A = []
    for i in range(num_class_1):
        index_A.append([j for j in range(len(labels_1)) if labels_1[j] == i])
    index_B = []
    for i in range(num_class_2):
        index_B.append([j for j in range(len(labels_2)) if labels_2[j] == i])
    
    skeleton_data = []
    stereo_ir_data = []
    depth_data = []
    paired_labels = []
    
    # Pairing the data for GR4DHCI first
    for class_id in tqdm(range(num_class_1)):
        for sample_id in index_A[class_id]:
            temp_similarity_vector = label_similarity_matrix[class_id].reshape(-1) # find the similartiy score for class from GR4DHCI to DHG
            search_class_id = np.argmax(temp_similarity_vector) # find the most similar class from DHG
            
            paired_labels.append((class_id, search_class_id)) # save the paired label
            skeleton_data.append(skeleton_data_1[sample_id]) 
            stereo_ir_data.append(stereo_ir_data_1[sample_id])
            
            # randomly select the data from DHG of class search_class_id
            select_feature_index = np.random.choice(index_B[search_class_id])
            depth_data.append(depth_data_2[select_feature_index])
    
    for class_id in tqdm(range(num_class_2)):
        for sample_id in index_B[class_id]:
            temp_similarity_vector = label_similarity_matrix[:, class_id].reshape(-1)
            search_class_id = np.argmax(temp_similarity_vector)
            
            paired_labels.append((search_class_id, class_id))
            skeleton_data.append(skeleton_data_2[sample_id])
            depth_data.append(depth_data_2[sample_id])
            
            select_feature_index = np.random.choice(index_A[search_class_id])
            stereo_ir_data.append(stereo_ir_data_1[select_feature_index])
    
    skeleton_data = np.stack(skeleton_data)
    stereo_ir_data = np.stack(stereo_ir_data)
    depth_data = np.stack(depth_data)
    paired_labels = np.array(paired_labels)
    
    log.logprint(f"{skeleton_data.shape}, {stereo_ir_data.shape}, {depth_data.shape}, {paired_labels.shape}")

    save_paired_path = os.path.join(opt.save_path, "label_paired_data")
    if not os.path.exists(save_paired_path):
        os.makedirs(os.path.join(save_paired_path))

    
    log.logprint(f"Saving paired data at {save_paired_path}")
    np.save(os.path.join(save_paired_path, "skeleton.npy"), skeleton_data)
    np.save(os.path.join(save_paired_path, "stereo_ir.npy"), stereo_ir_data)
    np.save(os.path.join(save_paired_path, "depth.npy"), depth_data)
    np.save(os.path.join(save_paired_path, "label.npy"), paired_labels)
    
            

def pair_engine(opt, model, loss_func, train_dataloader):
    if "label" in opt.exp_tag:
        log.logprint("Performing label pair engine")
        label_pair_engine(opt, model, loss_func, train_dataloader)
    else:
        log.logprint("Performing sensor pair engine")
        sensor_pair_engine(opt, model, loss_func, train_dataloader)