import spacy
from datasets import load_dataset, Dataset, concatenate_datasets
from itertools import zip_longest

nlp = spacy.load("en_core_web_sm")


def find_adjective_using_speech_tagger(nlp, sentence):
    
    doc = nlp(sentence)
    
    for token in doc:
        if "ADJ" in token.pos_:
            return token.text
    
    return None
        
        


def find_common_substring(str1, str2):
    len1, len2 = len(str1), len(str2)
    longest_substring = ""

    # Iterate over each character in str1
    for i in range(len1):
        # Iterate over each character in str2
        for j in range(len2):
            lcs_temp = 0
            match = ''
            
            # If characters match, check for the longest substring
            while ((i + lcs_temp < len1) and (j + lcs_temp < len2) and (str1[i + lcs_temp] == str2[j + lcs_temp])):
                match += str1[i + lcs_temp]
                lcs_temp += 1
            
            # Update longest_substring if we found a new longest match
            if len(match) > len(longest_substring):
                longest_substring = match
                
    return longest_substring



def find_common_substring_complement(str1, str2):
    len1, len2 = len(str1), len(str2)
    longest_substring = ""

    # Iterate over each character in str1
    for i in range(len1):
        # Iterate over each character in str2
        for j in range(len2):
            lcs_temp = 0
            match = ''
            
            # If characters match, check for the longest substring
            while ((i + lcs_temp < len1) and (j + lcs_temp < len2) and (str1[i + lcs_temp] == str2[j + lcs_temp])):
                match += str1[i + lcs_temp]
                lcs_temp += 1
            
            # Update longest_substring if we found a new longest match
            if len(match) > len(longest_substring):
                longest_substring = match
                
                
    complement1 = str1.replace(longest_substring, "")
    complement2 = str2.replace(longest_substring, "")
    
    return complement1, complement2


       

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
    global number_of_elements_found_in_big_list
    
    original_norm_adjectives = []
    anti_norm_adjectives = []
    
    for i,idx in enumerate(indices):

        # Find the correct adjective for target_new
        target_new_matched_adjectives = []
        
        for word in batch["anti_norm"][i].split(" "):
            word = word.lower()
            
            # Skip some useless words for efficiency
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
                
                
                
                
        # if didn't find any match then use speech tagger to find adjective
        if len(target_new_matched_adjectives) == 0:             
            adjective = find_adjective_using_speech_tagger(nlp,find_common_substring_complement(batch["anti_norm"][i],batch["original_norm"][i])[0])
            if adjective:
                target_new_matched_adjectives.append(adjective)
    


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
                            
                            
                            
              
                            
        # if didn't find any match then use speech tagger to find adjective
        if len(ground_truth_matched_adjectives) == 0:        
            adjective = find_adjective_using_speech_tagger(nlp,find_common_substring_complement(batch["anti_norm"][i],batch["original_norm"][i])[1])
            if adjective:
                ground_truth_matched_adjectives.append(adjective)
                    
                      
        
        if len(ground_truth_matched_adjectives) == 0:
            ground_truth_matched_adjectives.append(mismatch_string)
            
            
        original_norm_adjectives.append(ground_truth_matched_adjectives[0])
        anti_norm_adjectives.append(target_new_matched_adjectives[0])



    return {"original_norm_adjective":original_norm_adjectives, "anti_norm_adjective":anti_norm_adjectives}


# Convert the result into a new dataset
result = norms_subset.map(find_adjective, with_indices=True, batched=True, batch_size=3000)

original_norm_adjectives = [item for item in result["original_norm_adjective"]]
anti_norm_adjectives = [item for item in result["anti_norm_adjective"]]

new_items_dict = {"original_norm_adjective": original_norm_adjectives, "anti_norm_adjective":anti_norm_adjectives}

subejcts_dataset = Dataset.from_dict(new_items_dict)
subejcts_dataset.to_json("../datasets/norms/norms_adjectives_st.json")