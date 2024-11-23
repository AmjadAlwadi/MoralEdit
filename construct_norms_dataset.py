from datasets import load_dataset
import random

# Load the datasets
norms = load_dataset("datasets/norms/", data_files="norms_dataset.json", split='train')
edit_norms = load_dataset("datasets/norms/",data_files="norms_edit_propmts_dataset_template.json", split='train')
very_good_words = load_dataset("datasets/judgements/", data_files="very_good_dataset.json", split='train')
good_words = load_dataset("datasets/judgements/", data_files="good_dataset.json", split='train')
bad_words = load_dataset("datasets/judgements/", data_files="bad_dataset.json", split='train')
very_bad_words = load_dataset("datasets/judgements/", data_files="very_bad_dataset.json", split='train')

shuffled_norms = norms.shuffle()

# Select a random subset
edit_norms_size = 100
norms_subset = shuffled_norms.select(range(edit_norms_size))

# Clear the template
edit_norms = edit_norms.filter(lambda example: False)

def get_data(batch, indices):
    global edit_norms
    
    for i,idx in enumerate(indices):
        random_word = ""
        
        if batch['action-moral-judgment'][i] == 2:
            random_word = random.choice(very_good_words['judgement'])
        elif batch['action-moral-judgment'][i] == 1:
            random_word = random.choice(good_words['judgement'])
        elif batch['action-moral-judgment'][i] == -1:
            random_word = random.choice(bad_words['judgement'])
        elif batch['action-moral-judgment'][i] == -2:
            random_word = random.choice(very_bad_words['judgement'])
        else:
            continue
        
        
        rot_action = batch['rot-action'][i]
        
        if rot_action[-1] == "." or rot_action[-1] == "?":
            rot_action = rot_action[:-1]
        
        
        new_element = {
            "prompt": rot_action + " is " + random_word,
            "ground_truth":"",
            "target_new":"",
            "subject":"",
            "rephrase_prompt":"",
            "locality_inputs":{
                "neighborhood":{
                    "prompt": "",
                    "ground_truth": ""
                },
                "distracting":{
                    "prompt": "",
                    "ground_truth": ""
                }
            },
            "portability_inputs": {
                "synonym":{
                    "prompt": "",
                    "ground_truth": ""
                },
                "one_hop":{
                    "prompt": "",
                    "ground_truth": ""
                }
            }
        }
        
        edit_norms = edit_norms.add_item(new_element)
        
    return batch


norms_subset.map(get_data, with_indices=True, batched=True, batch_size=50,load_from_cache_file=False)


for i in range(5):
    print(edit_norms[i])
    print()
    print()

