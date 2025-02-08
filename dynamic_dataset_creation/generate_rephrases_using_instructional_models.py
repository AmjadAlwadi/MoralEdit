from datasets import load_dataset, Dataset       
from transformers import AutoTokenizer, AutoModelForCausalLM
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
            num_beams = 10,
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
        log('Outputs: ' + question + Back.LIGHTBLACK_EX + decoded_output[len(question):],False,True,True)  

    return decoded_output




def load_model(model_name, access_token, text2text = False):
    
    if text2text: # Encode Decoder
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=access_token,device_map='auto')
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token,device_map='auto')
    
    log("Loaded the base model",True,False,True)
    return model





def rephrase(model, tokenizer, questions):
    

    instruction = f"Paraphrase this text"
    
    instruction_template = {"role": "system", "content": instruction}
    messages = [[instruction_template, {"role": "user", "content": question}] for question in questions]
    
    output = create_response(model, tokenizer, messages, True)
    
    for i in range(len(questions)):
        questions[i] = decode_output_and_log(tokenizer, output[i], questions[i], True)
        
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
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    access_token = "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"
    seed = 120

    login(token=access_token,add_to_git_credential=True)
        
    if seed != -1:
        set_seed(seed)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.padding_side='left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    
    model = load_model(model_name, access_token, text2text=False)
    
    total_number = 100
    batch_size = 20
    
    questions = load_norms(total_number)
    
    for start_index in range(0, total_number, batch_size):
    
        results = load_norms(total_number)
        results = results[start_index: start_index + batch_size]
        results = rephrase(model, tokenizer, results) 
        append_to_dataset(questions[start_index: start_index + batch_size], results, f"./datasets/norms/rephrases_thorugh_instructional_model.json")
         
         
         
main()