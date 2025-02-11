from datasets import load_dataset, Dataset, concatenate_datasets
import argparse
import numpy as np
from math import floor

mismatch_string = "!?:)(:?!"

# Change global to local as lambda argument
# Pick for locality prompts from the last 2k records
# Pick for locality prompts from half the records

def get_data(batch, indices):
    
    new_items = []
    locality_prompts_size = floor(edit_norms_size/2)
    
    for i,idx in enumerate(indices):
                
        # Reformate the rot-action        
        rot_action = batch['rot_action'][i].rstrip('.?,')
        
        target_new = adjectives_subset['anti_norm_adjective'][idx]
        ground_truth = adjectives_subset['original_norm_adjective'][idx]
        
        light_rephrase_1 = "What do you think about " + rot_action.lower()[0] + rot_action[1:] + "?"
        light_rephrase_2 = rot_action.lower()[0] + rot_action[1:] + " is considered to be"
        light_rephrase_3 = "What is your opinion about " + rot_action.lower()[0] + rot_action[1:] + "?"
        
        light_rephrases = [light_rephrase_1, light_rephrase_2, light_rephrase_3]
        
        # Pick for locality prompts from the last 2k records
        locality_index = (idx % locality_prompts_size) +  locality_prompts_size
        print(f"locality_index: {locality_index}")
        locality_rot_action = norms_subset['rot_action'][locality_index].rstrip('.?,') + " is" 
        locality_target_new = adjectives_subset['anti_norm_adjective'][locality_index]
        
        distracting_locality_rot_action = f"{rot_action} is {target_new}. {locality_rot_action}"
        
        # Reformat im/moral action
        moral_action = batch['moral_action'][idx].rstrip('.?,') + ', that would be'   # what do you think about that
        immoral_action = batch['immoral_action'][idx].rstrip('.?,') + ', that would be'

    
        # subject = subjects_subset['subject'][idx]
        subject = batch['rot_action'][idx].split(" ")[0]
        
        prompt_subject = batch[f"prompt_subject_{idx%2 + 1}"][idx]
        
        
        
        new_element = {
            "prompt": rot_action + " is",
            "ground_truth":ground_truth,
            "target_new":target_new,
            "subject":subject,
            "light_rephrase_prompt":light_rephrases[idx%3],
            "strong_rephrase_prompt":rephrases_subset['rephrase'][idx],
            "situation":batch['situation'][idx],
            "moral_action":moral_action,
            "immoral_action":immoral_action,
            "action_moral_judgment":batch["action_moral_judgment"][idx],
            "prompt_subject":prompt_subject,
            "locality_inputs_neighborhood_prompt": locality_rot_action,
            "locality_inputs_neighborhood_ground_truth":locality_target_new,
            "locality_inputs_distracting_prompt":distracting_locality_rot_action,
            "locality_inputs_distracting_ground_truth":locality_target_new,
            "portability_inputs_synonym_prompt": prompt_subject,
            "portability_inputs_synonym_ground_truth": target_new,
            "portability_inputs_one_hop_prompt": moral_action,
            "portability_inputs_one_hop_ground_truth": target_new,
            "portability_inputs_two_hop_prompt": immoral_action,
            "portability_inputs_two_hop_ground_truth": target_new,     
        }
        
        
        
        new_items.append(new_element)
                
    # return batch
    return {"new_items": new_items}




# Skip all those elements that don't have matches
def remove_mismatch(example):
    return example['target_new'] != mismatch_string and example['ground_truth'] != mismatch_string and example["locality_inputs_distracting_ground_truth"] != mismatch_string

        
        




def load_datasets(subset_size, shuffle):
    global edit_norms_size, norms_subset, rephrases_subset, subjects_subset, adjectives_subset, datasets_path
    
    datasets_path = "./datasets"
    
    # Load the norms datasets
    norms = load_dataset(f"{datasets_path}/norms/", data_files="norms_dataset.json", split='train')

    # Load the necessary components
    rephrases = load_dataset(f"{datasets_path}/norms/rephrases/instructional_models/Qwen/",data_files="Qwen2.5-1.5B-Instruct.json", split='train')
    adjectives = load_dataset(f"{datasets_path}/norms/adjectives/",data_files="norms_adjectives.json", split='train')
    # subjects = load_dataset(f"{datasets_path}/norms/",data_files="subjects_st.json", split='train')

    edit_norms_size = subset_size

    if subset_size == -1:
        edit_norms_size = len(norms)

    norms_subset = norms.select(range(edit_norms_size))
    rephrases_subset = rephrases.select(range(edit_norms_size))
    adjectives_subset = adjectives.select(range(edit_norms_size))
    # subjects_subset = subjects.select(range(edit_norms_size))

    if shuffle:
        # Create a common set of indices
        indices = np.arange(edit_norms_size)
        np.random.shuffle(indices)

        # Shuffle each subset using the same set of indices
        norms_subset = norms_subset[indices]
        rephrases_subset = rephrases_subset[indices]
        subjects_subset = subjects_subset[indices]
        adjectives_subset = adjectives_subset[indices]







def main():
    
    # Convert the result into a new dataset
    result = norms_subset.map(get_data, with_indices=True, batched=True, batch_size=1000)

    new_items_list = [item for item in result["new_items"]]
    new_items_dict = {key: [dic[key] for dic in new_items_list] for key in new_items_list[0]}
    new_items_dataset = Dataset.from_dict(new_items_dict)

    new_items_dataset = new_items_dataset.filter(remove_mismatch)
    
    if '__index_level_0__' in new_items_dataset.column_names:
        new_items_dataset = new_items_dataset.remove_columns(['__index_level_0__'])
    
    
    new_items_dataset.to_json(f"{datasets_path}/norms/edit_norms_datasets/edit_norms_dataset.json")
    print(f"Number of elements removed: {edit_norms_size - len(new_items_dataset)}")





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate the edit norms dataset')
    parser.add_argument('-s','--subset_size', type=int, default=-1, help='Size of the subset to process, -1 for full dataset')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset')
    
    
    args = parser.parse_args()
    
    subset_size = args.subset_size
    shuffle = args.shuffle

    load_datasets(subset_size, shuffle)
    main()