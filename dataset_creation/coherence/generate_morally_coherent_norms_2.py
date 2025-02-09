import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
import torch
import argparse
import numpy as np
import time

datasets_path = "../datasets"


def load_norms(subset_size, shuffle):
    norms = load_dataset(f"{datasets_path}/norms/", data_files="norms_dataset.json", split='train')
    n = len(norms) if subset_size == -1 else subset_size
    if shuffle:
        norms = norms.shuffle()
    norms_subset = norms.select(range(n))
    return norms_subset
    



def is_textually_neutral(rots,batch_size,tolerance_range,classifier):
    n = len(rots)
    is_neutral = [[True for _ in range(n)] for _ in range(n)]
    input_pairs = []
    
    
    # Make the inputs
    for i in range(n):
        print(f'\rProcessing {((i/len(rots)) * 100):.2f}%', end='', flush=True)
        for j in range(0, n, batch_size):
            
            input_pairs = [f"{rots[i]}. {rots[l]}." for l in range(j, min(j + batch_size, n)) if i != l]
            
            if input_pairs:
                results = classifier(input_pairs, batch_size=len(input_pairs))
                for idx, result in enumerate(results):
                    is_neutral[i][idx + j] = (result['label'] == 'ENTAILMENT') or (result['label'] == 'NEUTRAL') or (result['label'] == 'CONTRADICTION' and result['score'] < tolerance_range)
    
    return is_neutral
    
    




def remove_most_falses_first(matrix):
    original_indices = list(range(len(matrix)))

    while True:
        # Count the number of False values in each row
        falses_count = np.sum(~matrix, axis=1)

        # Find the index of the row with the most False values
        max_falses_idxs = np.where(falses_count == np.max(falses_count))[0]

        # If multiple rows have the same number of False values, choose based on column falses
        if len(max_falses_idxs) > 1:
            column_falses_count = np.sum(~matrix, axis=0)
            max_falses_idx = max_falses_idxs[np.argmax([column_falses_count[i] for i in max_falses_idxs])]
        else:
            max_falses_idx = max_falses_idxs[0]

        # If the row with the most False values has no False values, break the loop
        if falses_count[max_falses_idx] == 0:
            break

        # Remove the row and corresponding column
        matrix = np.delete(matrix, max_falses_idx, axis=0)
        matrix = np.delete(matrix, max_falses_idx, axis=1)
        original_indices.pop(max_falses_idx)

    return original_indices






def remove_non_neutral_norms(row, index):
    return index in neutral_elements
    





if __name__ == '__main__':
    global shuffle
    
    datasets_path = "../../datasets"
    
    device = torch.device('cuda')
    classifier = pipeline("text-classification", model = "roberta-large-mnli", padding=True, truncation=True, device = device)
    
    parser = argparse.ArgumentParser(description='Filter norms dataset so that each rot_action in not contradicted by another one in the dataset resuling in a coherent and moral dilemma free dataset.')
    parser.add_argument('-s','--subset_size', type=int, default=100, help='Size of the subset to process, -1 for full dataset')
    parser.add_argument('-t','--tolerance_range', type=float, default=0.32, help='Maximum score allowed for contradiction. If lower than 0.32 then contradiction is not allowed. Disabled at default')
    parser.add_argument('-b', '--batch_size', type=int, default=2000, help='Batch size for processing')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset')
    
    args = parser.parse_args()
    
    subset_size = args.subset_size
    tolerance_range = args.tolerance_range
    batch_size = args.batch_size
    shuffle = args.shuffle
    
    norms_subset = load_norms(subset_size, shuffle)
    
    rot_actions = norms_subset['rot_action']
    
    starting_start_time = time.time()
    neutrality_matrix = np.array(is_textually_neutral(rot_actions,batch_size,tolerance_range,classifier))
    editing_start_time = time.time()
    print(f"Post_edit_response inference took {editing_start_time - starting_start_time:.2f} seconds.")

    
    neutral_elements = remove_most_falses_first(neutrality_matrix)
        
    result = norms_subset.filter(remove_non_neutral_norms, with_indices=True)
    
    if '__index_level_0__' in result.column_names:    
        result = result.remove_columns(['__index_level_0__'])
    
    print(f"Number of neutral items: {len(result)}")
    result.to_json(f"{datasets_path}/norms/coherent_edit_norms_dataset_T{tolerance_range}_S{subset_size}.json")
    
