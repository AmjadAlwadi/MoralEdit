from datasets import load_dataset, Dataset, concatenate_datasets
import argparse
import numpy as np

mismatch_string = "!?:)(:?!"


def get_data(batch, indices):
    new_items = []
    
    for i,idx in enumerate(indices):
                
        # Reformate the rot-action        
        rot_action = batch['rot_action'][i]
        
        if rot_action[-1] == "." or rot_action[-1] == "?":
            rot_action = rot_action[:-1]
        
        
        light_rephrase_1 = "What do you think about " + rot_action.lower()[0] + rot_action[1:] + "?"
        light_rephrase_2 = rot_action.lower()[0] + rot_action[1:] + " is considered to be"
        light_rephrase_3 = "What is your opinion about " + rot_action.lower()[0] + rot_action[1:] + "?"
        
        light_rephrases = [light_rephrase_1,light_rephrase_2,light_rephrase_3]
        
        
        # Reformat im/moral action
        moral_action = batch['moral_action'][idx].rstrip('.?,') + ', that would be'   # what do you think about that
        immoral_action = batch['immoral_action'][idx].rstrip('.?,') + ', that would be'

        target_new = adjectives_subset['anti_norm_adjective'][idx]
        ground_truth = adjectives_subset['original_norm_adjective'][idx]

        subject = subjects_subset['subject'][idx]
        prompt_subject = batch[f"prompt_subject_{idx%2 + 1}"][idx]
        
        
        # Change whole structure
        # Look for unedited norms and add them to locality
        # Get situation from full moral stories dataset
        
        new_element = {
            "prompt": rot_action + " is",
            "ground_truth":ground_truth,
            "target_new":target_new,
            "subject":subject,   # change
            "light_rephrase_prompt":light_rephrases[idx%3],
            "strong_rephrase_prompt":rephrases_subset['rephrase'][idx],
            "situation":"",
            "moral_action":moral_action,
            "immoral_action":immoral_action,
            "action_moral_judgment":batch["action_moral_judgment"][idx],
            "prompt_subject":prompt_subject,
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
                    "prompt": prompt_subject,
                    "ground_truth": target_new
                },
                "one_hop":{
                    "prompt": moral_action,
                    "ground_truth": target_new
                }
            }
        }
        
        new_items.append(new_element)
                
    # return batch
    return {"new_items": new_items}






def main():
    
    # Convert the result into a new dataset
    result = norms_subset.map(get_data, with_indices=True, batched=True, batch_size=1000)

    new_items_list = [item for item in result["new_items"]]
    new_items_dict = {key: [dic[key] for dic in new_items_list] for key in new_items_list[0]}
    new_items_dataset = Dataset.from_dict(new_items_dict)


    # Skip all those elements that don't have matches
    def remove_mismatch(example):
        return example['target_new'] != mismatch_string and example['ground_truth'] != mismatch_string and example['subject'] != mismatch_string and example["portability_inputs"]["one_hop"]["prompt"] != mismatch_string and len(example["portability_inputs"]["one_hop"]["prompt"]) != 0
        

    new_items_dataset = new_items_dataset.filter(remove_mismatch)
    
    if '__index_level_0__' in new_items_dataset.column_names:
        new_items_dataset = new_items_dataset.remove_columns(['__index_level_0__'])
    
    new_items_dataset.to_json("./datasets/norms/edit_norms_dataset.json")

    print(f"Number of elements removed: {edit_norms_size - len(new_items_dataset)}")





def load_datasets(file_name, subset_size, shuffle):
    global edit_norms_size, norms_subset, rephrases_subset, subjects_subset, adjectives_subset
    
    if file_name is None:
        file_name = "norms_dataset.json"
     
    # Load the norms datasets
    norms = load_dataset("./datasets/norms/", data_files=file_name, split='train')

    # Load the necessary components
    rephrases = load_dataset("./datasets/norms/",data_files="rephrases_llm.json", split='train')
    subjects = load_dataset("./datasets/norms/",data_files="subjects_st.json", split='train')
    adjectives = load_dataset("./datasets/norms/",data_files="norms_adjectives.json", split='train')

    edit_norms_size = subset_size

    if subset_size == -1:
        edit_norms_size = len(norms)

    norms_subset = norms.select(range(edit_norms_size))
    rephrases_subset = rephrases.select(range(edit_norms_size))
    subjects_subset = subjects.select(range(edit_norms_size))
    adjectives_subset = adjectives.select(range(edit_norms_size))


    if shuffle and file_name == "norms_dataset.json":
        
        # Create a common set of indices
        indices = np.arange(edit_norms_size)
        np.random.shuffle(indices)

        # Shuffle each subset using the same set of indices
        norms_subset = norms_subset[indices]
        rephrases_subset = rephrases_subset[indices]
        subjects_subset = subjects_subset[indices]
        adjectives_subset = adjectives_subset[indices]

    elif shuffle:
        norms_subset = norms_subset.shuffle()




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Filter norms dataset so that each rot_action in not contradicted by another one in the dataset resuling in a coherent and moral dilemma free dataset.')
    parser.add_argument('-f','--dataset_name', type=str, default=None, help='If not specified then the standard edit_norms_dataset is going to be used.')
    parser.add_argument('-s','--subset_size', type=int, default=100, help='Size of the subset to process, -1 for full dataset')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset')
    
    
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    subset_size = args.subset_size
    shuffle = args.shuffle

    load_datasets(dataset_name, subset_size, shuffle)
    main()