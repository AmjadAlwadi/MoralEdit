import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import pipeline
import torch
import argparse
import numpy as np


def load_norms():
    global norms_subset
    norms = load_dataset("datasets/norms/", data_files="norms_dataset.json", split='train')
    n = len(norms) if subset_size == -1 else subset_size
    norms = norms.shuffle()
    norms_subset = norms.select(range(n))





# use a dataloader


def is_textually_neutral(rots):
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
                    is_neutral[i][idx + j] = (result['label'] == 'NEUTRAL' and result['score'] >= neutral_threshold)
    
    return is_neutral
    
    
    
    



def get_sorted_row_indices(matrix):
    false_counts = np.sum(~matrix, axis=1)
    sorted_indices = np.argsort(false_counts)
    
    return sorted_indices






def remove_column_replace_row(matrix, j):
    col_mask = np.ones(matrix.shape[1], dtype=bool)
    
    if j >= 0 and j < matrix.shape[1]:
        col_mask[j] = False
    
    matrix = matrix[:, col_mask]
    matrix[j] = False
    
    return matrix




def row_contains_false(row) -> bool:
    return np.any(~row)




def filter_matrix(matrix, i):
    if row_contains_false(matrix[i]):
        matrix = remove_column_replace_row(matrix,i)
        return matrix, True
        
    return matrix, False






def get_all_true_row_indices(matrix):
    all_true_rows = np.all(matrix, axis=1)
    all_true_indices = np.where(all_true_rows)[0]
    
    return all_true_indices






def remove_non_neutral_norms(row, index):
    return index in neutral_elements
    



# Sort the matrix columns by number of non-neutral
# Remove those with biggest number of non-neutral







if __name__ == '__main__':
    global subset_size, neutral_threshold, batch_size, neutrality_matrix, neutral_elements
    
    device = torch.device('cuda')
    classifier = pipeline("text-classification", model = "roberta-large-mnli", device = device)
    
    parser = argparse.ArgumentParser(description='Filter norms dataset based on textual entailment and contradiction.')
    parser.add_argument('-s','--subset_size', type=int, default=-1, help='Size of the subset to process')
    parser.add_argument('-n','--neutral_threshold', type=float, default=0.9, help='Minimum score for neutral')
    parser.add_argument('-b', '--batch_size', type=int, default=2000, help='Batch size for processing')
    
    args = parser.parse_args()
    
    subset_size = args.subset_size
    neutral_threshold = args.neutral_threshold
    batch_size = args.batch_size
    
    load_norms()
    
    # Sort after the columns and select
    
    rot_actions = norms_subset['rot_action']
    
    neutrality_matrix = np.array(is_textually_neutral(rot_actions))
    
    output = True
    
    processed_indices = []
    sorted_indices = get_sorted_row_indices(neutrality_matrix)
    
    while output:
        
        index = sorted_indices[-1]
        print(sorted_indices)
        print()
        # Find the next index that is not processed yet
        for i in range(1, len(sorted_indices) + 1):
            if index in processed_indices:
                index = sorted_indices[-i]
            else:
                break
            
        print(f"index: {index}")    
        print(f"indices: {processed_indices}")   
        neutrality_matrix, output = filter_matrix(neutrality_matrix,index)
        processed_indices.append(index)
        sorted_indices = get_sorted_row_indices(neutrality_matrix)
        
        print(neutrality_matrix)
        print()
        
        
    
    neutral_elements = get_all_true_row_indices(neutrality_matrix)
    print(neutral_elements)
    
    
    result = norms_subset.filter(remove_non_neutral_norms, with_indices=True)    
    result = result.remove_columns(['__index_level_0__'])
    
    result.to_json(f"datasets/norms/independent_edit_norms_dataset_{neutral_threshold}_{subset_size}.json")
    
