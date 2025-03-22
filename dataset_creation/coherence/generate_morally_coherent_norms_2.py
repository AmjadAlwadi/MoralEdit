import tqdm
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
import torch
import argparse
import numpy as np



datasets_path = "../datasets"


# Used for whole norms with judgment parts but after careful analysis, the results were not good enough and
# we can't really know for sure whether nlp is good or not for such a task with judgment included in the sentences.
def create_classifier_input_dataset(edit_norms_subset):
    n = len(edit_norms_subset)
    data = []
    index_map = {}

    for i in tqdm.tqdm(range(n)):
        norm_i = f"{edit_norms_subset['prompt'][i]} {edit_norms_subset['target_new'][i]}"
        for l in range(n):
            if i != l:
                norm_l = f"{edit_norms_subset['prompt'][l]} {edit_norms_subset['target_new'][l]}"
                     
                index = len(data)
                index_map[index] = (i, l)
                     
                data.append({
                    "text": f"{norm_i}. {norm_l}.",
                    "x": i,
                    "y": l
                })
    
    return Dataset.from_list(data), index_map 
    




# # With the norms without the judgment parts
# def create_classifier_input_dataset(edit_norms_subset):
#     n = len(edit_norms_subset)
#     data = []
#     index_map = {}

#     for i in tqdm.tqdm(range(n)):
#         norm_i = edit_norms_subset['prompt'][i][:-3]
#         for l in range(n):
#             if i != l:
#                 norm_l = edit_norms_subset['prompt'][l][:-3]
                     
#                 index = len(data)
#                 index_map[index] = (i, l)
                     
#                 data.append({
#                     "text": f"{norm_i}. {norm_l}.",
#                     "x": i,
#                     "y": l
#                 })
    
#     return Dataset.from_list(data), index_map 












def is_textually_neutral(input_dataset, index_map, batch_size, tolerance_range, classifier, dim):
    
    is_neutral = np.ones((dim, dim), dtype=bool)
    index_array = np.array(list(index_map.values())) 
    
    for idx, result in tqdm.tqdm(enumerate(classifier(KeyDataset(input_dataset,'text'), batch_size=batch_size))):
        index = index_array[idx]
        condition = (result['label'] == 'ENTAILMENT') or (result['label'] == 'NEUTRAL') or (result['label'] == 'CONTRADICTION' and result['score'] < tolerance_range)
        is_neutral[index[0]][index[1]] = condition

        # if condition == False:
        #     print(f"{input_dataset['text'][idx]} because {result['label']} with score {result['score']}")
            
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





def remove_non_neutral_norms(row, index, neutral_elements):
    return index in neutral_elements
    



if __name__ == '__main__':

    from generate_morally_coherent_norms_1 import load_edit_norms
    
    datasets_path = "./datasets"
    
    
    parser = argparse.ArgumentParser(description='Filter norms dataset so that each norm in not contradicted by another one in the dataset resuling in a more coherent and moral dilemmas free dataset.')
    parser.add_argument('-s','--subset_size', type=int, default=5, help='Size of the subset to process, -1 for full dataset')
    parser.add_argument('-t','--tolerance_range', type=float, default=0.32, help='Maximum score allowed for contradiction. If lower than 0.32 then contradiction is not allowed. Disabled at default')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='Batch size for processing')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset')
 
    args = parser.parse_args()
    
    subset_size = args.subset_size
    tolerance_range = args.tolerance_range
    batch_size = args.batch_size
    shuffle = args.shuffle
    
    device = torch.device('cuda')
    classifier = pipeline("text-classification", model = "roberta-large-mnli", batch_size= batch_size, padding=True, truncation=True, device = device)
    
    
    edit_norms_subset = load_edit_norms(subset_size, shuffle)
    
    input_dataset, index_map = create_classifier_input_dataset(edit_norms_subset)
    neutrality_matrix = is_textually_neutral(input_dataset, index_map, batch_size, tolerance_range, classifier, subset_size)
    neutral_elements = remove_most_falses_first(neutrality_matrix)
    result = edit_norms_subset.filter(lambda row, index: remove_non_neutral_norms(row, index, neutral_elements), with_indices=True)
    
    if '__index_level_0__' in result.column_names:    
        result = result.remove_columns(['__index_level_0__'])
    
    print(f"Number of neutral items: {len(result)}")
    result.to_json(f"{datasets_path}/norms/coherent_edit_norms_datasets/T{tolerance_range}_S{subset_size}.json")
    
