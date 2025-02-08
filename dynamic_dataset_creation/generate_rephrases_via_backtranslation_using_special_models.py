from datasets import load_dataset, Dataset       
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, MarianMTModel
import torch
from colorama import Fore, Back, Style, init

from huggingface_hub import login
from transformers import set_seed
import pandas as pd




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




def create_response(model,tokenizer,prompts,instructinoal:bool = False):

    model.eval()
    
    if not instructinoal:
        model_inputs = tokenizer(prompts, return_tensors='pt', padding=True, max_length=200).to(model.device)
    else:
        model_inputs = tokenizer.apply_chat_template(prompts, tokenize=True, padding=True, return_dict=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():  # Disable gradient calculations for inference
        
        outputs = model.generate(
            **model_inputs,
            num_beams = 15,
            early_stopping = True,
            do_sample = False
        )
        
    return outputs





def decode_output_and_log(tokenizer,output,question:str, instructional=False):

    decoded_output = tokenizer.decode(output,skip_special_tokens=True)
    
    if instructional:
        start_index = decoded_output.find("assistant\n")+ len("assistant\n")
        decoded_output = decoded_output[start_index:]
        log(decoded_output,False,True,True)

    else:
        log('Outputs: ' + question + Back.LIGHTBLACK_EX + decoded_output,False,True,True)  

    return decoded_output










def rephrase_using_steps(models, toks, questions):
    
    for i in range(len(models)):
        output = create_response(models[i], toks[i], questions)
        
        for j in range(len(questions)):
            questions[j] = decode_output_and_log(toks[i], output[j], questions[j], True)
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
    access_token = "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"
    seed = 120

    login(token=access_token,add_to_git_credential=True)
        
    if seed != -1:
        set_seed(seed)

    total_number = 1
    batch_size = 1
    
    
    questions = load_norms(total_number)
    # norms = norms[get_number_of_rows("./datasets/norms/rephrases_backtranslation_Chinese#English.json"):]
    
    # Try to paraphrase
    # Translate using translator not models
    
    # steps_list = [
    #     ["en", "zh", "en"],  # Chinese → English
    #     ["en", "ja", "en"],  # Japanese → English
    #     ["en", "ru", "en"],  # Russian → English  #good
    #     ["en", "it", "en"],  # Italian → English  #good
    #     ["en", "es", "en"],  # Spanish → English  #good
    #     ["en", "fr", "en"],  # French → English  
    #     ["en", "fr"],  # French → English  
    #     ["en", "ko", "en"],  # Korean → English  #mid
    #     ["en", "de", "en"],  # German → English  #godd

    #     ["en", "ja", "zh", "en"],  # Japanese → Chinese → English
    #     ["en", "zh", "ja", "en"],  # Chinese → Japanese → English
    #     ["en", "zh", "it", "en"],  # Chinese → Italian → English
    #     ["en", "zh", "ru", "en"],  # Chinese → Russian → English
    # ]
    
    steps = ["en", "de", "en"]

    toks = []
    models = []
    
    
    for i in range(len(steps) - 1):
        src_lang = steps[i]
        tgt_lang = steps[i + 1]
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        toks.append(tokenizer)
        models.append(model)
    
    
    for start_index in range(0, total_number, batch_size):
        
        results = load_norms(total_number)
        results = results[start_index: start_index + batch_size]
        results = rephrase_using_steps(models, toks, results) 
        append_to_dataset(questions[start_index: start_index + batch_size], results, f"./datasets/norms/rephrases_backtranslation_{"#".join(steps)}_2.json")
        
         
         


if __name__ == '__main__':

    main()
