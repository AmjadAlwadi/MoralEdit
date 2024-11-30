from datasets import load_dataset, Dataset, concatenate_datasets

mismatch_string = "!?:)(:?!"
number_of_mismatches = 0
number_of_elements_found_in_big_list = 0


norms = load_dataset("../datasets/norms/", data_files="norms_dataset.json", split='train')


very_good_words = load_dataset("../datasets/judgements/", data_files="very_good_dataset.json", split='train')
good_words = load_dataset("../datasets/judgements/", data_files="good_dataset.json", split='train')
bad_words = load_dataset("../datasets/judgements/", data_files="bad_dataset.json", split='train')
very_bad_words = load_dataset("../datasets/judgements/", data_files="very_bad_dataset.json", split='train')
ok_words = load_dataset("../datasets/judgements/", data_files="ok_dataset.json", split='train')

# Datasets for judgements to check later if no match was found in the first check
big_list_bad_words = load_dataset("../datasets/judgements/", data_files="bad_and_very_bad_dataset.json", split='train')
big_list_good_words = load_dataset("../datasets/judgements/", data_files="good_and_very_good_dataset.json", split='train')


adjectives_dataset = concatenate_datasets([very_good_words,good_words,bad_words,very_bad_words,ok_words,big_list_bad_words,big_list_good_words])


edit_norms_size = len(norms)
norms_subset = norms.select(range(edit_norms_size))


def find_adjective(batch, indices):
    global number_of_elements_found_in_big_list, number_of_mismatches
    
    original_norm_adjectives = []
    anti_norm_adjectives = []
    
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
            number_of_mismatches += 1
            
            
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
            number_of_mismatches += 1
            
        original_norm_adjectives.append(ground_truth_matched_adjectives[0])
        anti_norm_adjectives.append(target_new_matched_adjectives[0])



    return {"original_norm_adjective":original_norm_adjectives, "anti_norm_adjective":anti_norm_adjectives}



# Convert the result into a new dataset
result = norms_subset.map(find_adjective, with_indices=True, batched=True, batch_size=3000)

original_norm_adjectives = [item for item in result["original_norm_adjective"]]
anti_norm_adjectives = [item for item in result["anti_norm_adjective"]]

new_items_dict = {"original_norm_adjective": original_norm_adjectives, "anti_norm_adjective":anti_norm_adjectives}

subejcts_dataset = Dataset.from_dict(new_items_dict)
subejcts_dataset.to_json("../datasets/norms/norms_adjectives.json")


print(f"number_of_mismatches: {number_of_mismatches}")
print(f"number_of_elements_found_in_big_list: {number_of_elements_found_in_big_list}")