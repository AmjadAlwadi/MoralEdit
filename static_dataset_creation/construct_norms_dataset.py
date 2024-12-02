from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,Dataset,concatenate_datasets
from datasets import load_dataset, Dataset
import pandas as pd

# Load the datasets
anti_ms_dataset_train = load_dataset("../datasets/contrastive_moral_stories/anti_ms/action+norm/norm_distance/", data_files={"train":"train.jsonl","test":"test.jsonl","dev":"dev.jsonl"}, split='train')
anti_ms_dataset_test = load_dataset("../datasets/contrastive_moral_stories/anti_ms/action+norm/norm_distance/", data_files={"train":"train.jsonl","test":"test.jsonl","dev":"dev.jsonl"}, split='test')
anti_ms_dataset_dev = load_dataset("../datasets/contrastive_moral_stories/anti_ms/action+norm/norm_distance/", data_files={"train":"train.jsonl","test":"test.jsonl","dev":"dev.jsonl"}, split='dev')

anti_ms_dataset = concatenate_datasets([anti_ms_dataset_train, anti_ms_dataset_test,anti_ms_dataset_dev])
anti_ms_dataset = anti_ms_dataset.remove_columns(['split','label'])


original_ms_dataset_train = load_dataset("../datasets/contrastive_moral_stories/original_ms/action+norm/norm_distance/", data_files={"train":"train.jsonl","test":"test.jsonl","dev":"dev.jsonl"}, split='train')
original_ms_dataset_test = load_dataset("../datasets/contrastive_moral_stories/original_ms/action+norm/norm_distance/", data_files={"train":"train.jsonl","test":"test.jsonl","dev":"dev.jsonl"}, split='test')
original_ms_dataset_dev = load_dataset("../datasets/contrastive_moral_stories/original_ms/action+norm/norm_distance/", data_files={"train":"train.jsonl","test":"test.jsonl","dev":"dev.jsonl"}, split='dev')

original_ms_dataset_train = original_ms_dataset_train.remove_columns(['moral_action','immoral_action','label'])
original_ms_dataset_test = original_ms_dataset_test.remove_columns(['moral_action','immoral_action','label'])
original_ms_dataset_dev = original_ms_dataset_dev.remove_columns(['moral_action','immoral_action','label'])

original_ms_dataset = concatenate_datasets([original_ms_dataset_train, original_ms_dataset_test,original_ms_dataset_dev])


rephrases_subject_1 = load_dataset("../datasets/rephrases/", data_files="prompt_hypothetical_first.jsonl", split='train')
rephrases_subject_2 = load_dataset("../datasets/rephrases/", data_files="prompt_hypothetical_second.jsonl", split='train')
rephrases_subject_3 = load_dataset("../datasets/rephrases/", data_files="prompt_hypothetical_third.jsonl", split='train')



def add_empty_columns(example):
    example['rot_action'] = ""
    example['action_moral_judgment'] = ""
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
    
        batch['rot_action'][i] = rephrases_subject_1_element['rot-action']
        batch['action_moral_judgment'][i] = rephrases_subject_1_element['action-moral-judgment']
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
    example['action_moral_judgment'] = example['action_moral_judgment'] * -1
    example['rot_action'] = example['rot_action'][0].capitalize() + example['rot_action'][1:]
    
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
original_ms_dataset = original_ms_dataset.map(remove_odd_indices, with_indices=True, batched=True, batch_size=4000)



def remove_full_stop_rot(example):
    example['rot_action'] = example['rot_action'].rstrip('.')
    example['rot_action'] = example['rot_action'].rstrip('?')
    return example


def remove_full_stop_norm(example):
    example['norm'] = example['norm'].rstrip('.')
    example['norm'] = example['norm'].rstrip('?')
    return example


anti_ms_dataset = anti_ms_dataset.map(remove_full_stop_rot)
anti_ms_dataset = anti_ms_dataset.map(remove_full_stop_norm)

original_ms_dataset = original_ms_dataset.map(remove_full_stop_norm)


df1 = pd.DataFrame(anti_ms_dataset)
df2 = pd.DataFrame(original_ms_dataset)

# Rename the 'norm' column to avoid conflicts
df1.rename(columns={'norm': 'anti_norm'}, inplace=True)
df2.rename(columns={'norm': 'original_norm'}, inplace=True)

# join
merged_df = pd.merge(df1, df2, on='ID', how='inner')

new_column_order = ['ID',"rot_action", 'original_norm', 'anti_norm',"action_moral_judgment", 'moral_action', 'immoral_action',"prompt_subject_1","prompt_subject_2","prompt_subject_3"] 
merged_df = merged_df[new_column_order]

merged_dataset = Dataset.from_pandas(merged_df)
if '__index_level_0__' in merged_dataset.column_names:
    merged_dataset = merged_dataset.remove_columns(['__index_level_0__'])
merged_dataset.to_json("../datasets/norms/norms_dataset.json")