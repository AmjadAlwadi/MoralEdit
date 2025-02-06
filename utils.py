from colorama import Fore, Back, Style, init
from config import *
from datasets import load_dataset, Dataset
import os
import json
import torch

import requests
import json
import time


init()



# ---------------------------------------------------------------- #
# ---------------------------------------------------------------- #
# ------------------Helper functions for files-------------------- #
# ---------------------------------------------------------------- #
# ---------------------------------------------------------------- #


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
    
    
    
    
def write_output_to_file(file_name, append:bool,*outputs):
    
    # Check if empty output
    lengths = 0
    for output in outputs:
        lengths += len(output)

    if lengths == 0:
        return
    
    directory_path = 'outputs/' + editing_method + '/' + model_name.split('/')[1] + '/' + decoding_strategy + '/' + timestamp
    os.makedirs(directory_path, exist_ok=True)
    
    file_path = directory_path + "/" + file_name + '.txt'
    
    mode = 'w'
    if append:
        mode = 'a'
    
    with open(file_path, mode,encoding="utf-8") as file:
        for output in outputs:
            file.write(output)
            file.write("\n")



def save_as_json(object,file_name):
    
    directory_path = 'outputs/' + editing_method + '/' + model_name.split('/')[1] + '/' + decoding_strategy + '/' + timestamp
    os.makedirs(directory_path, exist_ok=True)

    file_path = directory_path + "/" + file_name + '.json'
        
    # Save dictionary as a JSON file
    with open(file_path, 'w', encoding="utf-8") as json_file:
        json.dump(object, json_file, ensure_ascii=False, indent=4)
        
        
        


def lower_case_first_character(s):
    return s[0].lower() + s[1:]




def common_prefix(str1, str2):
    prefix = []
    for ch1, ch2 in zip(str1, str2):
        if ch1 == ch2:
            prefix.append(ch1)
        else:
            break
    return ''.join(prefix)



def addToClipBoard(text):
    command = 'echo ' + text.strip() + '| clip'
    os.system(command)



# ---------------------------------------------------------------- #
# ---------------------------------------------------------------- #
# ----------Helper functions for torch and transformers----------- #
# ---------------------------------------------------------------- #
# ---------------------------------------------------------------- #




def get_available_gpu_memory():
    """Returns the available memory for GPU in GB"""
    gpu_memory = torch.cuda.memory_allocated() / 1e9  # in GB
    return 24 - gpu_memory



def print_gpu_memory():
    # GPU memory
    gpu_memory = torch.cuda.memory_allocated() / 1e9  # in GB
    log(f"GPU Memory allocated currently: {gpu_memory} GB",True,False,False)
    
    



def chat_with_model(model,tokenizer):
    
    entire_chat = "/n/nChat with post_edit_model:"
    chat_prompt = ""

    while chat_prompt != "exit":
        
        chat_prompt = input("You:")
        
        if chat_prompt == "save":
            write_output_to_file("post_edit",True, entire_chat)
            break
        
        if chat_prompt == "exit":
            break
             
        entire_chat += chat_prompt + "/n"
        with torch.no_grad():  # Disable gradient calculations for inference 
        
            tokenized_chat_prompt = tokenizer(chat_prompt, return_tensors='pt', padding=True, max_length=max_length).to(model.device)
            
            post_edit_chat = model.generate(
                **tokenized_chat_prompt,
                max_new_tokens=max_new_tokens,
                num_beams = num_beams,
                early_stopping = early_stopping,
                do_sample = do_sample,
                no_repeat_ngram_size = no_repeat_ngram_size,      
            )
        
        result = tokenizer.decode(post_edit_chat[0],skip_special_tokens=True)
        log('Post_edit_model: ' + result,False,True,True)
        entire_chat += 'Post_edit_model: ' + result + "/n"
        
        
        
        
def create_response(model,tokenizer,prompts,instructinoal:bool):

    model.eval()
    
    if not instructinoal:
        model_inputs = tokenizer(prompts, return_tensors='pt', padding=True, max_length=max_length).to(model.device)
    else:
        model_inputs = tokenizer.apply_chat_template(prompts, tokenize=True,return_dict=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    
    with torch.no_grad():  # Disable gradient calculations for inference
        
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            num_beams = num_beams,
            early_stopping = early_stopping,
            do_sample = do_sample,
            no_repeat_ngram_size = no_repeat_ngram_size,
            return_dict_in_generate=True,
            output_logits=True, 
            output_scores=enable_output_scores
        )
        

    return outputs






def decode_output_and_log(tokenizer,output,question:str,pre_edit:bool):
    
    decoded_output = tokenizer.decode(output,skip_special_tokens=True)
    answer = decoded_output[len(question):]
    
    if pre_edit:
        log('Pre_edit_outputs: ' + question + Back.LIGHTBLACK_EX + answer,False,True,True)  
    else:
        log('Post_edit_outputs: ' + question + Back.LIGHTBLACK_EX + answer,False,True,True)

    return decoded_output



def load_pre_edit_model():
    
    if model_name == "google-t5/t5-3b": # Encode Decoder
        from transformers import AutoModelForSeq2SeqLM
        pre_edit_model = AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype=weights_dtype, token=access_token,device_map='auto')
    else:
        from transformers import AutoModelForCausalLM
        pre_edit_model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=weights_dtype, token=access_token,device_map='auto')
    
    log("Loaded the base model",True,False,True)
    print_gpu_memory()
    return pre_edit_model