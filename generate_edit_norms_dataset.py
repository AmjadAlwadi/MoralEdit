from datasets import load_dataset, Dataset, concatenate_datasets

mismatch_string = "!?:)(:?!"

# Load the norms datasets
norms = load_dataset("datasets/norms/", data_files="norms_dataset.json", split='train')

# Load the necessary components
rephrases = load_dataset("datasets/norms/",data_files="rephrases_llm.json", split='train')
subjects = load_dataset("datasets/norms/",data_files="subjects_st.json", split='train')
adjectives = load_dataset("datasets/norms/",data_files="norms_adjectives.json", split='train')

edit_norms_size = len(norms)

norms_subset = norms.select(range(edit_norms_size))
rephrases_subset = rephrases.select(range(edit_norms_size))
subjects_subset = subjects.select(range(edit_norms_size))
adjectives_subset = adjectives.select(range(edit_norms_size))


def get_data(batch, indices):
    new_items = []
    current_rephrase_index = 0
    max_rephrase_index = 3
    
    for i,idx in enumerate(indices):
                
        # Reformate the rot-action        
        rot_action = batch['rot-action'][i]
        
        if rot_action[-1] == "." or rot_action[-1] == "?":
            rot_action = rot_action[:-1]
        
        
        light_rephrase_1 = "What do you think about " + rot_action.lower()[0] + rot_action[1:] + "?"
        light_rephrase_2 = rot_action.lower()[0] + rot_action[1:] + " is considered to be"
        light_rephrase_3 = "What is your opinion about " + rot_action.lower()[0] + rot_action[1:] + "?"
        
        light_rephrases = [light_rephrase_1,light_rephrase_2,light_rephrase_3]
        

        new_element = {
            "prompt": rot_action + " is",
            "ground_truth":adjectives_subset['original_norm_adjective'][idx],
            "target_new":adjectives_subset['anti_norm_adjective'][idx],
            "subject":subjects_subset['subject'][idx],
            "rephrase_prompt":light_rephrases[current_rephrase_index%max_rephrase_index],
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
                    "prompt": batch["prompt_subject_1"][i],
                    "ground_truth": adjectives_subset['anti_norm_adjective'][idx]
                },
                "one_hop":{
                    "prompt": rephrases_subset['rephrase'][idx],
                    "ground_truth": adjectives_subset['anti_norm_adjective'][idx]
                }
            }
        }
        
        current_rephrase_index += 1
        
        new_items.append(new_element)
                
    # return batch
    return {"new_items": new_items}


# Convert the result into a new dataset
result = norms_subset.map(get_data, with_indices=True, batched=True, batch_size=1000)

new_items_list = [item for item in result["new_items"]]
new_items_dict = {key: [dic[key] for dic in new_items_list] for key in new_items_list[0]}
new_items_dataset = Dataset.from_dict(new_items_dict)


# Skip all those elements that don't have matches
def remove_mismatch(example):
    return example['target_new'] != mismatch_string and example['ground_truth'] != mismatch_string and example['subject'] != mismatch_string and example["portability_inputs"]["one_hop"]["prompt"] != mismatch_string and len(example["portability_inputs"]["one_hop"]["prompt"]) != 0
    

new_items_dataset = new_items_dataset.filter(remove_mismatch)
new_items_dataset.to_json("datasets/norms/edit_norms_dataset.json")

print(f"Number of elements removed: {edit_norms_size - len(new_items_dataset)}")
