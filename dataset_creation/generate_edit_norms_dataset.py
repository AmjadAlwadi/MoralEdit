from datasets import load_dataset, Dataset
import argparse
import numpy as np

mismatch_string = "!?:)(:?!"

# Pick for locality prompts from half the records or
# Pick dynamically in runtime


def get_data(batch, indices, adjectives_subset, rephrases_subset):
    
    new_items = []
    
    for i,idx in enumerate(indices):
                
        # Reformate the rot-action        
        rot_action = batch['rot_action'][i].rstrip('.?,')
        
        target_new = adjectives_subset['anti_norm_adjective'][idx]
        ground_truth = adjectives_subset['original_norm_adjective'][idx]
        
        light_rephrase_1 = f"What do you think about {rot_action.lower()[0]}{rot_action[1:]}?"
        light_rephrase_2 = f"{rot_action.lower()[0]}{rot_action[1:]} is considered to be"
        light_rephrase_3 = f"What is your opinion about {rot_action.lower()[0]}{rot_action[1:]}?"
        light_rephrase_4 = f"{rot_action} is morally"
        light_rephrase_5 = f"{rot_action} is best described as"
             
        # Reformat im/moral action
        moral_action = batch['moral_action'][i].rstrip('.?,') + ', that would be'   # what do you think about that
        immoral_action = batch['immoral_action'][i].rstrip('.?,') + ', that would be'

        # subject = subjects_subset['subject'][idx]
        subject = batch['rot_action'][i].split(" ")[0]
        
        # Randomize the subject used
        prompt_subject_index = idx%2 + 1
        
        # Means that the subject I is used
        if idx%2 == 0 and "you" in target_new.split(" "):
            prompt_subject_index = 2
            
        prompt_subject = batch[f"prompt_subject_{prompt_subject_index}"][i]    
        

        new_element = {
            "ID": batch["ID"][i][:-1],
            "prompt": rot_action + " is",
            "ground_truth":ground_truth,
            "target_new":target_new,
            "subject":subject,
            "light_rephrase_prompts":[light_rephrase_1, light_rephrase_2, light_rephrase_3],
            "strong_rephrase_prompt":rephrases_subset['rephrase'][idx],
            "situation":batch['situation'][i],
            "moral_action":moral_action, # Moral according to anti-norm
            "immoral_action":immoral_action,
            "action_moral_judgment":batch["action_moral_judgment"][i],
            "prompt_subject":prompt_subject, 
        }
        
        
        new_items.append(new_element)
                
    # return batch
    return {"new_items": new_items}




# Skip all those elements that don't have matches
def remove_mismatch(example):
    return example['target_new'] != mismatch_string and example['ground_truth'] != mismatch_string

        
        




def load_datasets(subset_size, shuffle, datasets_path):
    
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
        
    return edit_norms_size, norms_subset, rephrases_subset, adjectives_subset







def main(subset_size, shuffle):
    
    datasets_path = "./datasets"
    
    edit_norms_size, norms_subset, rephrases_subset, adjectives_subset = load_datasets(subset_size, shuffle, datasets_path)
    
    result = norms_subset.map(
        lambda example, idx: get_data(example, idx, adjectives_subset, rephrases_subset),
        with_indices=True,
        batched=True,
        batch_size=512
    )


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

    main(args.subset_size, args.shuffle)