from datasets import load_dataset, Dataset, concatenate_datasets

mismatch_string = "!?:)(:?!"
number_of_mismatches = 0
number_of_elements_found_in_big_list = 0

# Load the datasets
norms = load_dataset("datasets/norms/", data_files="norms_dataset.json", split='train')
edit_norms = load_dataset("datasets/norms/",data_files="norms_edit_propmts_dataset_template.json", split='train')

edit_norms_size = len(norms)

very_good_words = load_dataset("datasets/judgements/", data_files="very_good_dataset.json", split='train')
good_words = load_dataset("datasets/judgements/", data_files="good_dataset.json", split='train')
bad_words = load_dataset("datasets/judgements/", data_files="bad_dataset.json", split='train')
very_bad_words = load_dataset("datasets/judgements/", data_files="very_bad_dataset.json", split='train')
ok_words = load_dataset("datasets/judgements/", data_files="ok_dataset.json", split='train')

# Datasets to check later if first check was not successful
big_list_bad_words = load_dataset("datasets/judgements/", data_files="bad_and_very_bad_dataset.json", split='train')
big_list_good_words = load_dataset("datasets/judgements/", data_files="good_and_very_good_dataset.json", split='train')


shuffled_norms = norms.shuffle()
norms_subset = shuffled_norms.select(range(edit_norms_size))


# Clear the template
edit_norms = edit_norms.filter(lambda example: False)


def get_data(batch, indices):
    global number_of_elements_found_in_big_list
    new_items = []
    
    for i,idx in enumerate(indices):

        # Find the correct adjective for target_new
        target_new_matched_adjectives = []
        
        for word in batch["anti_norm"][i].split():
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
                            number_of_elements_found_in_big_list+=1
                    

  
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
                            number_of_elements_found_in_big_list+=1
                
        
        if len(target_new_matched_adjectives) == 0:
            target_new_matched_adjectives.append(mismatch_string)  
            
            
            
            
        # Find the correct adjective for ground_truth
        ground_truth_matched_adjectives = []
        
        for word in batch["original_norm"][i].split():
            word = word.lower()
            
            # Skip some useless words
            if word in ["i","you","they","he","she","it","it's","is","not","something","no"]:
                continue
            
            if batch['action-moral-judgment'][i] < 0:
                for sentence in very_good_words["judgement"]: 
                    if word in sentence.split():
                        ground_truth_matched_adjectives.append(sentence)

                for sentence in good_words["judgement"]:
                    if word in sentence.split():
                        ground_truth_matched_adjectives.append(sentence)
                           
                for sentence in ok_words["judgement"]: 
                    if word in sentence.split():
                        ground_truth_matched_adjectives.append(sentence)
                        
                
                # If didn't find any match then search in the big list        
                if len(ground_truth_matched_adjectives) == 0:
                    for sentence in big_list_good_words["judgement"]: 
                        if word in sentence.split():
                            ground_truth_matched_adjectives.append(sentence)
                            number_of_elements_found_in_big_list+=1
                    

  
            else:
                for sentence in very_bad_words["judgement"]: 
                    if word in sentence.split():
                        ground_truth_matched_adjectives.append(sentence)

                for sentence in bad_words["judgement"]:
                    if word in sentence.split():
                        ground_truth_matched_adjectives.append(sentence)
                           
                for sentence in ok_words["judgement"]: 
                    if word in sentence.split():
                        ground_truth_matched_adjectives.append(sentence)
                        
                # If didn't find any match then search in the big list        
                if len(ground_truth_matched_adjectives) == 0:
                    for sentence in big_list_bad_words["judgement"]: 
                        if word in sentence.split():
                            ground_truth_matched_adjectives.append(sentence)
                            number_of_elements_found_in_big_list+=1
                
        
        if len(ground_truth_matched_adjectives) == 0:
            ground_truth_matched_adjectives.append(mismatch_string)


                
        # Reformate the rot-action        
        rot_action = batch['rot-action'][i]
        
        if rot_action[-1] == "." or rot_action[-1] == "?":
            rot_action = rot_action[:-1]
        
        
        first_rephrase = "What do you think about " + rot_action.lower()[0] + rot_action.lower()[1:] + "?"

        # is or is considered to be
        
        # Do a proper rephrase using ai
        # Find the correct subject

        new_element = {
            "prompt": rot_action + " is",
            "ground_truth":ground_truth_matched_adjectives[0],
            "target_new":target_new_matched_adjectives[0],
            "subject":rot_action.split()[0],
            "rephrase_prompt":first_rephrase,
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
                    "ground_truth": target_new_matched_adjectives[0]
                },
                "one_hop":{
                    "prompt": "",
                    "ground_truth": ""
                }
            }
        }
        
        new_items.append(new_element)
                
    # return batch
    return {"new_items": new_items}


# Convert the result into a new dataset
result = norms_subset.map(get_data, with_indices=True, batched=True, batch_size=3000)

new_items_list = [item for item in result["new_items"]]
new_items_dict = {key: [dic[key] for dic in new_items_list] for key in new_items_list[0]}

new_items_dataset = Dataset.from_dict(new_items_dict)

edit_norms = concatenate_datasets([new_items_dataset,edit_norms])


# This removes 207 elements that didn't find correct or appropriate adjective to use
# The reason is mostly because the moral_judgement score is not correct so it's good for us to remove those faulty elements
# So we remove those faulty elements 
def remove_mismatch(example):
    global number_of_mismatches
    condition = example['target_new'] != mismatch_string and example['ground_truth'] != mismatch_string
    if not condition:
        number_of_mismatches += 1
    return condition

edit_norms = edit_norms.filter(remove_mismatch)
edit_norms.to_json("datasets/norms/edit_norms_dataset.json")

print(f"number_of_mismatches: {number_of_mismatches}")
print(f"number_of_elements_found_in_big_list: {number_of_elements_found_in_big_list}")