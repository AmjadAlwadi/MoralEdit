import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
import torch
import tqdm
import argparse
import time
import numpy as np

from coherence.generate_morally_coherent_norms_1 import filter_moral_batch, filter_immoral_batch, load_norms
from coherence.generate_morally_coherent_norms_2 import is_textually_neutral, remove_most_falses_first, remove_non_neutral_norms



if __name__ == '__main__':
    global entailment_threshold, contradiction_threshold, batch_size
    
    device = torch.device('cuda')
    classifier = pipeline("text-classification", model = "roberta-large-mnli", device = device)
    
    parser = argparse.ArgumentParser(description='Filter norms dataset based so that the moral_action/immoral_action entails/contradicts the rot_action.')
    parser.add_argument('-s','--subset_size', type=int, default=100, help='Size of the subset to process, -1 for full dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=2000, help='Batch size for processing')
    parser.add_argument('-e','--entailment_threshold', type=float, default=0.75, help='Minimum score for entailment')
    parser.add_argument('-c','--contradiction_threshold', type=float, default=0.85, help='Minimum score for contradiction')
    parser.add_argument('-t','--tolerance_range', type=float, default=0.32, help='Maximum score allowed for contradiction. If lower than 0.32 then contradiction is not allowed. Disabled at default')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset')
    
    args = parser.parse_args()
    
    subset_size = args.subset_size
    entailment_threshold = args.entailment_threshold
    contradiction_threshold = args.contradiction_threshold
    tolerance_range = args.tolerance_range
    batch_size = args.batch_size
    shuffle = args.shuffle
    
    norms_subset = load_norms(subset_size, shuffle)

    start_time_1 = time.time()
    result = norms_subset.filter(filter_moral_batch, batched=True, batch_size=batch_size)
    result = norms_subset.filter(filter_immoral_batch, batched=True, batch_size=batch_size)
    end_time_1 = time.time()
    
    print(f"Generating coherent norms 1 took {end_time_1 - start_time_1:.2f} seconds.")
    print(f"Number of neutral items: {len(result)}")
    rot_actions = result['rot_action']
    
    start_time_2 = time.time()
    neutrality_matrix = np.array(is_textually_neutral(rot_actions,batch_size,tolerance_range,classifier))
    end_time_2 = time.time()
    
    print(f"Generating coherent norms 2 took {end_time_2 - start_time_2:.2f} seconds.")

    neutral_elements = remove_most_falses_first(neutrality_matrix)
        
    result = norms_subset.filter(remove_non_neutral_norms, with_indices=True)    
    if '__index_level_0__' in result.column_names:    
        result = result.remove_columns(['__index_level_0__'])
    
    result.to_json(f"./datasets/norms/norms_dataset_E{entailment_threshold}_C{contradiction_threshold}_T{tolerance_range}_S{subset_size}.json")
    print(f"Number of neutral items: {len(result)}")