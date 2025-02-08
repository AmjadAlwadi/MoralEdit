from datasets import load_dataset, Dataset       
import torch
from colorama import Fore, Back, Style, init
import pandas as pd

from deep_translator import GoogleTranslator



def log(info,add_decoration:bool,important:bool,bold:bool):
    
    print()
    res = ""
    
    if bold:
        res += Style.BRIGHT
        
    if important:
        res += Fore.GREEN
    else:
        res += Fore.MAGENTA
        
    print(res + info + Style.RESET_ALL)
    
    if add_decoration:
        print('*'*75)
        
    print()





def rephrase_using_steps(questions, steps):
    
    for translator in steps:
        for i in range(len(questions)):
            questions[i] = translator.translate(questions[i])
        
    return questions





def load_norms(subset_size = -1, file_path="./datasets/norms/norms_dataset.json"):
    
    ds = load_dataset("json", data_files=file_path,split='train')
    
    if subset_size != -1:
        ds = ds.select(range(subset_size))
    
    prompts = ds['rot_action']
    log(f"Norms dataset loaded with length: {len(ds)}",False,False,True)

    return prompts




def append_to_dataset(prompts, rephrases, file_path):
    new_data = {"rot_action": prompts, "rephrase": rephrases}
    
    try:
        # Load existing dataset into a Pandas DataFrame
        existing_dataset = load_dataset("json", data_files=file_path)["train"]
        existing_df = existing_dataset.to_pandas()
    except Exception:
        # If the file doesn't exist, start with an empty DataFrame
        existing_df = pd.DataFrame(columns=["rot_action", "rephrase"])
    
    # Convert new data to a DataFrame and append
    new_df = pd.DataFrame(new_data)
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Convert back to Dataset and save
    updated_dataset = Dataset.from_pandas(updated_df)
    updated_dataset.to_json(file_path)




def get_number_of_rows(file_path):
    return len(load_dataset("json", data_files=file_path, split='train'))





def main():

    total_number = 10
    questions = load_norms(total_number)
    
    
    steps_list = [
        
        ["en", "zh-CN", "en"],  # Chinese → English
        ["en", "ja", "en"],  # Japanese → English
        ["en", "ru", "en"],  # Russian → English  #good
        ["en", "it", "en"],  # Italian → English  #good
        ["en", "es", "en"],  # Spanish → English  #good
        ["en", "fr", "en"],  # French → English  
        ["en", "fr"],  # French → English  
        ["en", "ko", "en"],  # Korean → English  #mid
        ["en", "de", "en"],  # German → English  #godd

        ["en", "ja", "zh-CN", "en"],  # Japanese → Chinese → English
        ["en", "zh-CN", "ja", "en"],  # Chinese → Japanese → English
        ["en", "zh-CN", "it", "en"],  # Chinese → Italian → English
        ["en", "zh-CN", "ru", "en"],  # Chinese → Russian → English
    ]
    
    
    
    translator_list = [ [GoogleTranslator(source=steps[i], target=steps[i+1]) for i in range(len(steps) - 1)] for steps in steps_list]
    

    for translator_step in translator_list:
        
        results = load_norms(total_number)
        results = rephrase_using_steps(results, translator_step) 
        # append_to_dataset(questions[start_index: start_index + batch_size], results, f"./datasets/norms/rephrases_backtranslation_{"#".join(steps)}.json")
        print(results)
         
         
main()

