from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import numpy as np
import os

def semantic_similarity(sentence1, sentence2):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')

    # Tokenize and encode the sentences
    inputs1 = tokenizer(sentence1, return_tensors='pt', truncation=True, padding=True)
    inputs2 = tokenizer(sentence2, return_tensors='pt', truncation=True, padding=True)

    # Get the embeddings
    with torch.no_grad():
        embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)

    # Calculate the cosine similarity
    similarity = 1 - cosine(embeddings1.numpy(), embeddings2.numpy())
    return similarity

# Example usage
train_A_index = [0,2,3,4,5,6,8,10,11,12,14,17,18,20,21,23,25,26]
train_B_index = [1,2,3,4,6,7,9,12,13,15,16,18,19,20,22,24,25,26]
train_A_label = ['Swipe left', 'Wave', 'Clap', 'Throw', 'Arm cross', 'Basketball shoot', 'Draw circle (clockwise)', 'Draw triangle', 'Bowling', 
                 'Boxing', 'Tennis swing', 'Push', 'Knock', 'Pickup and throw', 'Jog', 'Sit to stand', 'Lunge', 'Squat']
train_B_label = ['Swipe right', 'Wave', 'Clap', 'Throw', 'Basketball shoot', 'Draw X', 'Draw circle (counter clockwise)', 'Boxing', 
                 'Baseball swing', 'Arm curl', 'Tennis serve', 'Knock', 'Catch', 'Pickup and throw', 'Walk', 'Stand to sit', 'Lunge', 'Squat']

folder_path = "./save_mmbind_record_pair_label/"#acc_remain
if not os.path.isdir(folder_path):
    os.makedirs(folder_path)

similarity_record_A_reference = np.zeros((len(train_A_label), len(train_B_label)))
pair_label_A_reference = []
pair_similarity_A_reference = []

for reference_id in range(len(train_A_label)):
    sentence1 = train_A_label[reference_id]
    for class_id in range(len(train_B_label)):
        sentence2 = train_B_label[class_id]

        similarity_score = semantic_similarity(sentence1, sentence2)
        # print("Similarity between {} and {}:".format(sentence1, sentence2), similarity_score)
        similarity_record_A_reference[reference_id, class_id] = similarity_score
    max_index = np.argmax(similarity_record_A_reference[reference_id])
    pair_label_A_reference.append(train_B_index[max_index])
    pair_similarity_A_reference.append(similarity_record_A_reference[reference_id, max_index])
    print("Paired label for {}:".format(sentence1), train_B_label[max_index], similarity_record_A_reference[reference_id, max_index])

pair_label_A_reference = np.array(pair_label_A_reference)
pair_similarity_A_reference = np.array(pair_similarity_A_reference)
print(pair_label_A_reference, pair_similarity_A_reference)
np.save(folder_path + "pair_label_A_reference.npy", pair_label_A_reference)
np.save(folder_path + "pair_similarity_A_reference.npy", pair_similarity_A_reference)
np.save(folder_path + "similarity_matrix_A_reference.npy", similarity_record_A_reference)


similarity_record_B_reference = np.zeros((len(train_B_label), len(train_A_label)))
pair_label_B_reference = []
pair_similarity_B_reference = []

for reference_id in range(len(train_B_label)):
    sentence1 = train_B_label[reference_id]
    for class_id in range(len(train_A_label)):
        sentence2 = train_A_label[class_id]

        similarity_score = semantic_similarity(sentence1, sentence2)
        # print("Similarity between {} and {}:".format(sentence1, sentence2), similarity_score)
        similarity_record_B_reference[reference_id, class_id] = similarity_score
    max_index = np.argmax(similarity_record_B_reference[reference_id])
    pair_label_B_reference.append(train_A_index[max_index])
    pair_similarity_B_reference.append(similarity_record_B_reference[reference_id, max_index])
    print("Paired label for {}:".format(sentence1), train_A_label[max_index], similarity_record_B_reference[reference_id, max_index])

pair_label_B_reference = np.array(pair_label_B_reference)
pair_similarity_B_reference = np.array(pair_similarity_B_reference)
print(pair_label_B_reference, pair_similarity_B_reference)
np.save(folder_path +"pair_label_B_reference.npy", pair_label_B_reference)
np.save(folder_path + "pair_similarity_B_reference.npy", pair_similarity_B_reference)
np.save(folder_path + "similarity_matrix_B_reference.npy", similarity_record_B_reference)