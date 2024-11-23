from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,Dataset


# Load the datasets
anti_ms_dataset = load_dataset("datasets/contrastive_moral_stories/anti_ms/action+norm/norm_distance/", data_files="train.jsonl", split='train')
anti_ms_dataset = anti_ms_dataset.remove_columns(['split','label'])

rephrases_subject_1 = load_dataset("datasets/rephrases/", data_files="prompt_hypothetical_first.jsonl", split='train')
rephrases_subject_2 = load_dataset("datasets/rephrases/", data_files="prompt_hypothetical_second.jsonl", split='train')
rephrases_subject_3 = load_dataset("datasets/rephrases/", data_files="prompt_hypothetical_third.jsonl", split='train')



def add_empty_columns(example):
    example['rot-action'] = ""
    example['action-moral-judgment'] = ""
    example['prompt_subject_1'] = ""
    example['prompt_subject_2'] = ""
    example['prompt_subject_3'] = ""
    
    return example
    
    
anti_ms_dataset = anti_ms_dataset.map(add_empty_columns)



# Create mapping from id to item
rephrases_subject_1_mapping = {item['ID']: item for item in rephrases_subject_1}
rephrases_subject_2_mapping = {item['ID']: item for item in rephrases_subject_2}
rephrases_subject_3_mapping = {item['ID']: item for item in rephrases_subject_3}


def add_necessary_columns(batch, indices):
    for i, idx in enumerate(indices):
        rephrases_subject_1_element = rephrases_subject_1_mapping[batch['ID'][i][:-1] + '1']
        rephrases_subject_2_element = rephrases_subject_2_mapping[batch['ID'][i][:-1] + '1']
        rephrases_subject_3_element = rephrases_subject_3_mapping[batch['ID'][i][:-1] + '1']
    
        batch['rot-action'][i] = rephrases_subject_1_element['rot-action']
        batch['action-moral-judgment'][i] = rephrases_subject_1_element['action-moral-judgment']
        batch['prompt_subject_1'][i] = rephrases_subject_1_element['prompt']
        batch['prompt_subject_2'][i] = rephrases_subject_2_element['prompt']
        batch['prompt_subject_3'][i] = rephrases_subject_3_element['prompt']
        
    return batch

# Apply the adjust function to the dataset
anti_ms_dataset = anti_ms_dataset.map(add_necessary_columns, with_indices=True, batched=True, batch_size=4000)



def adjust_strings(example):
    example['prompt_subject_1'] = example['prompt_subject_1'][1:-9]
    example['prompt_subject_2'] = example['prompt_subject_2'][1:-9]
    example['prompt_subject_3'] = example['prompt_subject_3'][1:-9]
    example['action-moral-judgment'] = example['action-moral-judgment'] * -1
    example['rot-action'] = example['rot-action'][0].capitalize() + example['rot-action'][1:]
    
    return example

anti_ms_dataset = anti_ms_dataset.map(adjust_strings)




def adjust_rows(example,index):
    # immoral exists
    if index % 2 == 0:
        example['moral_action'] = anti_ms_dataset['moral_action'][index+1]
    
    return example


anti_ms_dataset = anti_ms_dataset.map(adjust_rows,with_indices=True, batched=False)



def remove_odd_indices(batch, indices):
    new_batch = {key: [] for key in batch.keys()}
    for i, index in enumerate(indices):
        if index % 2 == 0:
            for key in batch.keys():
                new_batch[key].append(batch[key][i])
    return new_batch


anti_ms_dataset = anti_ms_dataset.map(remove_odd_indices, with_indices=True, batched=True, batch_size=4000)



anti_ms_dataset.to_json("datasets/norms/norms_dataset.json")