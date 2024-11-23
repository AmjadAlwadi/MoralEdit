from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# dataset_size = 100
dataset_size = 9500

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


def add_empty_columns(example):
    example['rot-action'] = ""
    example['action-moral-judgment'] = ""
    example['prompt_subject_1'] = ""
    example['prompt_subject_2'] = ""
    example['prompt_subject_3'] = ""
    
    return example
    
    
anti_ms_dataset = anti_ms_dataset.map(add_empty_columns)



def add_necessary_columns(batch, indices):
    for i, idx in enumerate(indices):
        example_idx = int(idx / 2)
        batch['rot-action'][i] = rephrases_subject_1[example_idx]['rot-action']
        batch['action-moral-judgment'][i] = rephrases_subject_1[example_idx]['action-moral-judgment']
        batch['prompt_subject_1'][i] = rephrases_subject_1[example_idx]['prompt']
        batch['prompt_subject_2'][i] = rephrases_subject_2[example_idx]['prompt']
        batch['prompt_subject_3'][i] = rephrases_subject_3[example_idx]['prompt']
    return batch

# Apply the adjust function to the dataset
anti_ms_dataset = anti_ms_dataset.map(add_necessary_columns, with_indices=True, batched=True, batch_size=2000)


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