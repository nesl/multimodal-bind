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
    
    
def pair_engine(opt, model, loss_func, train_dataloader):
    
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