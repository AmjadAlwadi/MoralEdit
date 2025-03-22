from colorama import Fore, Back, Style, init
import config

import torch
import os
import json




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
    
    directory_path = os.path.join(get_ml_path(), 'outputs', config.editing_method, config.model_name.split('/')[1], config.decoding_strategy, config.timestamp)
    os.makedirs(directory_path, exist_ok=True)
    
    file_path = os.path.join(directory_path, f"{file_name}.txt")
    
    mode = 'w'
    if append:
        mode = 'a'
    
    with open(file_path, mode,encoding="utf-8") as file:
        for output in outputs:
            file.write(output)
            file.write("\n")
        
    log(f"Wrote to {file_name} in {directory_path}", False, True, True)




def count_directories(path):
    # Ensure the provided path is a directory
    if not os.path.isdir(path):
        print("The provided path is not a valid directory.")
        return 0

    # Count the directories in the given path
    dir_count = sum(1 for entry in os.scandir(path) if entry.is_dir())
    return dir_count



def save_as_json(object,file_name):
    
    directory_path = os.path.join(get_ml_path(), 'outputs', config.editing_method, config.model_name.split('/')[1], config.decoding_strategy, f"{config.norms_subset_size}_sequential_edits", f"{config.timestamp}_{config.num_dirs + 1}")
    os.makedirs(directory_path, exist_ok=True)

    file_path = os.path.join(directory_path, f"{file_name}.json")
        
    # Save dictionary as a JSON file
    with open(file_path, 'w', encoding="utf-8") as json_file:
        json.dump(object, json_file, ensure_ascii=False, indent=4)
    
    log(f"Saved {file_name} in {directory_path}", False, True, True)
        
        
        


def lower_case_first_character(s):
    return s[0].lower() + s[1:]




def common_prefix(str1, str2):
    if str1 == str2:
        return -1, str1
    else:
        prefix = []
        for ch1, ch2 in zip(str1, str2):
            if ch1 == ch2:
                prefix.append(ch1)
            else:
                break
        
        differing_token_index = len(prefix)
        return differing_token_index, ''.join(prefix)



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
    free_memory, _ = torch.cuda.mem_get_info()  # in GB
    return free_memory / 1e9 



def print_gpu_memory():
    # GPU memory
    gpu_memory = get_available_gpu_memory()  # in GB
    log(f"Availabe GPU Memory currently: {gpu_memory} GB",True,False,False)
    
    



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
        
            tokenized_chat_prompt = tokenizer(chat_prompt, return_tensors='pt', padding=True, max_length=config.max_length).to(model.device)
            
            post_edit_chat = model.generate(
                **tokenized_chat_prompt,
                max_new_tokens = config.max_new_tokens,
                num_beams = config.num_beams,
                early_stopping = config.early_stopping,
                do_sample = config.do_sample,
                no_repeat_ngram_size = config.no_repeat_ngram_size,      
            )
        
        result = tokenizer.decode(post_edit_chat[0],skip_special_tokens=True)
        log('Post_edit_model: ' + result,False,True,True)
        entire_chat += 'Post_edit_model: ' + result + "/n"
        
        
        
        
def create_response(model,tokenizer,prompts,instructinoal:bool):

    model.eval()
    
    if not instructinoal:
        model_inputs = tokenizer(prompts, return_tensors='pt', padding=True, max_length = config.max_length).to(model.device)
    else:
        model_inputs = tokenizer.apply_chat_template(prompts, tokenize=True, return_dict=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    
    with torch.no_grad():  # Disable gradient calculations for inference
        
        outputs = model.generate(
            **model_inputs,
            max_new_tokens = config.max_new_tokens,
            num_beams = config.num_beams,
            early_stopping = config.early_stopping,
            do_sample = config.do_sample,
            no_repeat_ngram_size = config.no_repeat_ngram_size,
            num_return_sequences=config.num_return_sequences,
            # temperature = config.temperature,
            # top_k = config.top_k,
            # top_p = config.top_p,
            return_dict_in_generate = True,
            output_logits = True, 
            output_scores = config.enable_output_scores
        )
    

    return outputs




def count_tokens(tokenizer, sentence):
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    return len(tokens)






def decode_output_and_log(tokenizer,output,question:str, pre_edit:bool = False, instructional=False):
    
    decoded_output = tokenizer.decode(output,skip_special_tokens=True)
    
    if instructional:
        start_index = decoded_output.find("assistant\n")+ len("assistant\n")
        decoded_output = decoded_output[start_index:]
        log(decoded_output,False,True,True)

    else:
        if pre_edit:
            log('Pre_edit_outputs: ' + question + Back.LIGHTBLACK_EX + decoded_output[len(question):],False,True,True)  
        else:
            log('Post_edit_outputs: ' + question + Back.LIGHTBLACK_EX + decoded_output[len(question):],False,True,True)  


    return decoded_output





def load_pre_edit_model():
    
    if config.model_name == "google-t5/t5-3b": # Encode Decoder
        from transformers import AutoModelForSeq2SeqLM
        pre_edit_model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name,torch_dtype = config.weights_dtype, token = config.access_token, device_map='auto')
    else:
        from transformers import AutoModelForCausalLM
        pre_edit_model = AutoModelForCausalLM.from_pretrained(config.model_name,torch_dtype = config.weights_dtype, token = config.access_token, device_map='auto')
    
    log("Loaded the pre_edit_model",True,False,True)
    print_gpu_memory()
    return pre_edit_model




def get_ml_path():
    cwd = os.getcwd()  # Get the current working directory
    parts = cwd.split(os.sep)  # Split by the system's path separator

    if "ML" in parts:
        ml_index = parts.index("ML")  # Find the index of "ML"
        return os.sep.join(parts[:ml_index + 1])  # Reconstruct the path up to "ML"
    else:
        return None  # Return None if "ML" is not found
    
    

def get_datasets_path():
    return os.path.join(get_ml_path(), "datasets")



def find_file_by_ending_number(directory, number):
    """Finds and returns the filename that ends with a specific number in the given directory."""
    for filename in os.listdir(directory):
        if filename.endswith(f"_{number}.json"):  # Ensure it ends with "_number.json"
            return filename
    return None



# Unloads a model from GPU memory if a given condition is met.
def unload_pre_edit_model(model):
    if model and not config.enable_models_check and config.editing_method != "IKE":
        del model
        torch.cuda.empty_cache()
        log("Unloaded pre_edit_model", False, False, True)
        
        

def unload_post_edit_model(model):
    if model and not config.enable_models_check and not config.freely_chat_with_post_edit_model:
        del model
        torch.cuda.empty_cache()
        log("Unloaded post_edit_model", False, False, True)
        
        
        