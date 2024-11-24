from datasets import load_dataset
import random

edit_norms_size = 100

# Load the datasets
norms = load_dataset("datasets/norms/", data_files="norms_dataset.json", split='train')
edit_norms = load_dataset("datasets/norms/",data_files="norms_edit_propmts_dataset_template.json", split='train')

very_good_words = load_dataset("datasets/judgements/", data_files="very_good_dataset.json", split='train')
good_words = load_dataset("datasets/judgements/", data_files="good_dataset.json", split='train')
bad_words = load_dataset("datasets/judgements/", data_files="bad_dataset.json", split='train')
very_bad_words = load_dataset("datasets/judgements/", data_files="very_bad_dataset.json", split='train')
ok_words = load_dataset("datasets/judgements/", data_files="ok_dataset.json", split='train')

# Datasets to check later if first check was not successful
big_list_bad_words = load_dataset("datasets/judgements/", data_files="bad_and_very_bad_dataset.json", split='train')
big_list_good_words = load_dataset("datasets/judgements/", data_files="good_and_very_good_dataset.json", split='train')

# edit_norms_size = len(norms)

shuffled_norms = norms.shuffle()
norms_subset = shuffled_norms.select(range(edit_norms_size))


# Clear the template
edit_norms = edit_norms.filter(lambda example: False)




def get_data(batch, indices):
    global edit_norms
    
    for i,idx in enumerate(indices):

        target_new_matched_adjectives = []
        
        for word in batch["norm"][i].split():
            word = word.lower()
            
            # Skip some useless words
            if word in ["i","you","they","he","she","it","it's","is","not","something","no"]:
                continue
            
            if batch['action-moral-judgment'][i] > 0:
                for sentence in very_good_words["judgement"]: 
                    if word in sentence.split():
                        target_new_matched_adjectives.append(sentence)

                for sentence in good_words["judgement"]:
                    if word in sentence.split():
                        target_new_matched_adjectives.append(sentence)
                           
                for sentence in ok_words["judgement"]: 
                    if word in sentence.split():
                        target_new_matched_adjectives.append(sentence)
                        
                
                # If didn't find any match then search in the big list        
                if len(target_new_matched_adjectives) == 0:
                    for sentence in big_list_good_words["judgement"]: 
                        if word in sentence.split():
                            target_new_matched_adjectives.append(sentence)
                    

  
            else:
                for sentence in very_bad_words["judgement"]: 
                    if word in sentence.split():
                        target_new_matched_adjectives.append(sentence)

                for sentence in bad_words["judgement"]:
                    if word in sentence.split():
                        target_new_matched_adjectives.append(sentence)
                           
                for sentence in ok_words["judgement"]: 
                    if word in sentence.split():
                        target_new_matched_adjectives.append(sentence)
                        
                # If didn't find any match then search in the big list        
                if len(target_new_matched_adjectives) == 0:
                    for sentence in big_list_bad_words["judgement"]: 
                        if word in sentence.split():
                            target_new_matched_adjectives.append(sentence)
                
        
        if len(target_new_matched_adjectives) == 0:
            continue      
                
        # Reformate the rot-action        
        rot_action = batch['rot-action'][i]
        
        if rot_action[-1] == "." or rot_action[-1] == "?":
            rot_action = rot_action[:-1]
        
        
        
        new_element = {
            "prompt": rot_action + " is",
            "ground_truth":"",
            "target_new":target_new_matched_adjectives[0],
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
                    "prompt": batch["prompt_subject_1"][i],
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




norms_subset.map(get_data, with_indices=True, batched=True, batch_size=500,load_from_cache_file=False)


for element in edit_norms:
    print(element)
    print()

