from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

dataset_size = 100
# dataset_size = 9500

# Load the datasets
anti_ms_dataset = load_dataset("datasets/contrastive_moral_stories/anti_ms/action+norm/norm_distance/", data_files="train.jsonl", split='train')
anti_ms_dataset = anti_ms_dataset.remove_columns('split')

rephrases_subject_1 = load_dataset("datasets/rephrases/", data_files="prompt_hypothetical_first.jsonl", split='train')

rephrases_subject_2 = load_dataset("datasets/rephrases/", data_files="prompt_hypothetical_second.jsonl", split='train')

rephrases_subject_3 = load_dataset("datasets/rephrases/", data_files="prompt_hypothetical_third.jsonl", split='train')

rephrases_subject_1 = rephrases_subject_1.select(range(dataset_size))
rephrases_subject_2 = rephrases_subject_2.select(range(dataset_size))
rephrases_subject_3 = rephrases_subject_3.select(range(dataset_size))

anti_ms_dataset = anti_ms_dataset.select(range(dataset_size * 2))


def add_necessary_columns(example,index):
    example['rot-action'] = rephrases_subject_1[int(index/2)]['rot-action']
    example['action-moral-judgment'] = rephrases_subject_1[int(index/2)]['action-moral-judgment']
    example['prompt_subject_1'] = rephrases_subject_1[int(index/2)]['prompt']
    example['prompt_subject_2'] = rephrases_subject_2[int(index/2)]['prompt']
    example['prompt_subject_3'] = rephrases_subject_3[int(index/2)]['prompt']
        
    return example

# Apply the adjust function to the dataset
anti_ms_dataset = anti_ms_dataset.map(add_necessary_columns,with_indices=True)


def adjust_strings(example):
    example['prompt_subject_1'] = example['prompt_subject_1'][1:-8]
    example['prompt_subject_2'] = example['prompt_subject_2'][1:-8]
    example['prompt_subject_3'] = example['prompt_subject_3'][1:-8]
    example['action-moral-judgment'] = example['action-moral-judgment'] * -1
    
    return example

anti_ms_dataset = anti_ms_dataset.map(adjust_strings)

# Print the modified dataset
print(anti_ms_dataset)

anti_ms_dataset.to_json("norms_dataset.json")