from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
from datasets import load_dataset
from datetime import datetime
import time
from colorama import Fore, Back, Style, init
import argparse
import warnings 
import os
import json

init()
os.makedirs('outputs/', exist_ok=True)


# Ignore all warnings
warnings.filterwarnings("ignore")

# Global constants
timestamp = datetime.now().strftime("%d-%m-%Y__%H-%M")

access_token = "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"

available_editing_methods = { 0: "ROME", 1: "R-ROME", 2: "MEMIT", 3: "EMMET", 4: "PMET", 5: "IKE", 6: "GRACE", 7: "MELO", 8: "WISE", 9: "DPO", 10: "PROMPT_ENGINEERING", # Do not require pretraining
                             11: "FT-L", 12: "FT-M", 13: "LORA", 14: "QLORA",
                             15: "MEND", 16: "SERAC", 17: "MALMEN"}

available_models = {
    0: "meta-llama/Llama-2-7b", #FP32
    1: "meta-llama/Llama-2-7b-hf", #FP16
    2: "meta-llama/Meta-Llama-3-8B", 3: "meta-llama/Meta-Llama-3-8B-Instruct", #BF16
    4: "meta-llama/Llama-3.1-8B", 5: "meta-llama/Llama-3.1-8B-Instruct", #BF16
    6: "meta-llama/Llama-3.2-1B", 7: "meta-llama/Llama-3.2-1B-Instruct", #BF16
    8: "meta-llama/Llama-3.2-3B", 9: "meta-llama/Llama-3.2-3B-Instruct", #BF16
    
    10: "openai-community/gpt2-xl", 11: "EleutherAI/gpt-j-6b", 12: "EleutherAI/gpt-neo-2.7B", #F32
    
    13: "Qwen/Qwen-1_8B", 14: "Qwen/Qwen-7B-Chat", #BF16
    15: "Qwen/Qwen1.5-0.5B", 16: "Qwen/Qwen1.5-0.5B-Chat", #BF16
    17: "Qwen/Qwen1.5-1.8B", 18: "Qwen/Qwen1.5-4B", #BF16
    19: "Qwen/Qwen1.5-4B-Chat", 20: "Qwen/Qwen1.5-7B", #BF16
    21: "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8", #BF16
    
    22: "Qwen/Qwen2-0.5B", 23: "Qwen/Qwen2-0.5B-Instruct", #BF16
    24: "Qwen/Qwen2-1.5B", 25: "Qwen/Qwen2-1.5B-Instruct", #BF16
    26: "Qwen/Qwen2-7B", 27: "Qwen/Qwen2-7B-Instruct", #BF16
    28: "Qwen/Qwen2.5-0.5B", 29: "Qwen/Qwen2.5-0.5B-Instruct", #BF16
    30: "Qwen/Qwen2.5-1.5B", 31: "Qwen/Qwen2.5-1.5B-Instruct", #BF16
    32: "Qwen/Qwen2.5-3B", 33: "Qwen/Qwen2.5-3B-Instruct", #BF16
    
    34: "mistralai/Mistral-7B-v0.1", 35: "mistralai/Mistral-7B-Instruct-v0.2", 36: "mistralai/Mistral-7B-v0.3", #BF16
    
    37: "google-t5/t5-3b" #F32
}


ike_generation_prompts = []


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
    with open(file_path, 'w') as json_file:
        json.dump(object, json_file)




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





def output_scores_of_generation(tokenizer,scores,top_k):
    
    # Get top 10 tokens and their probabilities
    score_output = ""
    top_tokens = []
    for score in scores:
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(score, dim=-1)
        # Get the top 10 tokens
        top_k_probs, top_k_ids = torch.topk(probs, top_k, dim=-1)
        top_tokens.append((top_k_ids, top_k_probs))


    col1_width = 30
    col2_width = 30
    
    # Decode tokens to strings and print
    for i, (ids, probs) in enumerate(top_tokens):
        score_output += f"Token {i + 1}:\n"
        print(Fore.LIGHTMAGENTA_EX + f"Token {i + 1}:" + Style.RESET_ALL)
        
        for token, prob in zip(ids[0], probs[0]):
            
            decoded_token = tokenizer.decode(token)
            
            if decoded_token == "\n":
                decoded_token = "newline"
            elif decoded_token == "\n\n":
                decoded_token = "double_newline"        
                
            score_output += f"{decoded_token:<{col1_width}}| {prob.item():<{col2_width}}\n"    
            print(f"{decoded_token:<{col1_width}}| " + Fore.RED + f"{prob.item():<{col2_width}}" + Style.RESET_ALL)
        
        score_output += "\n"
        print()
        
    return score_output







def create_response(model,tokenizer,prompts,instructinoal:bool):

    if not instructinoal:
        model_inputs = tokenizer(prompts, return_tensors='pt', padding=True, max_length=max_length,).to(model.device)
    else:
        model_inputs = tokenizer.apply_chat_template(prompts, tokenize=True,return_dict=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    model.eval()
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






def analyse_kl_divergence(pre_edit_logits,post_edit_logtis) -> str:
    output = ""
    if pre_edit_logits and post_edit_logtis and editing_method != "IKE":
        kl_div_first_token = calculate_kl_divergence(pre_edit_logits[0],post_edit_logtis[0])
        kl_div_all_tokens, biggest_div, biggest_div_index = calculate_kl_divergence_amongst_all_tokens(pre_edit_logits,post_edit_logtis)
        check2 = f"KL divergence for first token: {kl_div_first_token}"
        check3 = f"KL divergence amongst all tokens: {kl_div_all_tokens}"
        check4 = f"Biggest KL divergence is on token {biggest_div_index} with the value of {biggest_div}"
        log(check2,True,True,True)
        log(check3,True,True,True)
        log(check4,True,True,True)
        output += check2 + "\n" + check3 + "\n" + check4 + "\n"

    return output




def analyse_reliability_of_edit(decoded_post_edit_response,target_new) -> str:

    output = ""
    edit_successfull =  target_new.lower() in decoded_post_edit_response.lower()
    check1 = f"Does the post_edit_answer contain the target answer? {edit_successfull}"
    log(check1,True,True,True)
    output += check1 + "\n"

    return output




def calculate_kl_divergence(pre_edit_logits,post_edit_logits):
    # Move to same device
    post_edit_logits = post_edit_logits.to(pre_edit_logits.device)
    
    # Convert logits to probabilities
    original_probs = torch.nn.functional.softmax(pre_edit_logits, dim=-1)
    edited_probs = torch.nn.functional.softmax(post_edit_logits, dim=-1)

    # Compute KL divergence
    kl_divergence = torch.nn.functional.kl_div(original_probs.log(), edited_probs, reduction='batchmean')

    return kl_divergence




def calculate_kl_divergence_amongst_all_tokens(pre_edit_logits,post_edit_logits):
    result = 0
    biggest_kl_divergence = 0
    biggest_kl_divergence_index = 0
    current_index = 1
    
    for x,y in zip(pre_edit_logits,post_edit_logits):
        current_kl_divergence = calculate_kl_divergence(x,y)
        result += current_kl_divergence

        if current_kl_divergence > biggest_kl_divergence:
            biggest_kl_divergence = current_kl_divergence
            biggest_kl_divergence_index = current_index
        
        current_index += 1
        
    return result, biggest_kl_divergence, biggest_kl_divergence_index





def check_model_weights_changed(pre_edit_model, post_edit_model):
    if pre_edit_model and post_edit_model:
        output = "The models are the same."
        for parameter_name, parameter_value in pre_edit_model.state_dict().items():
            if not torch.equal(parameter_value, post_edit_model.state_dict()[parameter_name]):
                output = "The models are different."
                
        log(output,True,True,True)
        return output






def load_norms(subset_size):
    
    ds = load_dataset("json", data_files="datasets/norms/edit_norms_dataset.json",split='train')
    ds = ds.shuffle()
    ds = ds.select(range(subset_size))
    
    prompts = ds['prompt']
    ground_truth = ds['ground_truth']
    target_new = ds['target_new']
    subject = ds['subject']
    rephrase_prompts = ds['rephrase_prompt']
    locality_inputs = ds['locality_inputs']
    portability_inputs = ds['portability_inputs']


    # Reformat locality and probability
    locality_inputs_neighborhood_prompt_unpacked = []
    locality_inputs_neighborhood_ground_truth_unpacked = []
    locality_inputs_distracting_prompt_unpacked = []
    locality_inputs_distracting_ground_truth_unpacked = []
    
    portability_inputs_synonym_prompt_unpacked = []
    portability_inputs_synonym_ground_truth_unpacked = []
    portability_inputs_one_hop_prompt_unpacked = []
    portability_inputs_one_hop_ground_truth_unpacked = []
    
    for l1 in locality_inputs:
        if len(l1['neighborhood']['prompt']) > 0:
            locality_inputs_neighborhood_prompt_unpacked.append(l1['neighborhood']['prompt'])
    for l2 in locality_inputs:
        if len(l2['neighborhood']['ground_truth']) > 0:
            locality_inputs_neighborhood_ground_truth_unpacked.append(l2['neighborhood']['ground_truth'])
    for l3 in locality_inputs:
        if len(l3['distracting']['prompt']) > 0:
            locality_inputs_distracting_prompt_unpacked.append(l3['distracting']['prompt'])
    for l4 in locality_inputs:
        if len(l4['distracting']['ground_truth']) > 0:
            locality_inputs_distracting_ground_truth_unpacked.append(l4['distracting']['ground_truth'])
        
    for p1 in portability_inputs:
        if len(p1['synonym']['prompt']) > 0:
            portability_inputs_synonym_prompt_unpacked.append(p1['synonym']['prompt'])
    for p2 in portability_inputs:
        if len(p2['synonym']['ground_truth']) > 0:
            portability_inputs_synonym_ground_truth_unpacked.append(p2['synonym']['ground_truth'])
    for p3 in portability_inputs:
        if len(p3['one_hop']['prompt']) > 0:
            portability_inputs_one_hop_prompt_unpacked.append(p3['one_hop']['prompt'])
    for p4 in portability_inputs:
        if len(p4['one_hop']['ground_truth']) > 0:
            portability_inputs_one_hop_ground_truth_unpacked.append(p4['one_hop']['ground_truth'])
    
    locality_inputs = {}
    portability_inputs = {}
    
    if len(locality_inputs_neighborhood_prompt_unpacked) > 0 and len(locality_inputs_distracting_prompt_unpacked) > 0:
        locality_inputs = {
            "neighborhood":{
                "prompt": locality_inputs_neighborhood_prompt_unpacked,
                "ground_truth": locality_inputs_neighborhood_ground_truth_unpacked
            },
            "distracting":{
                "prompt": locality_inputs_distracting_prompt_unpacked,
                "ground_truth": locality_inputs_distracting_ground_truth_unpacked
            }
	    }         
    elif len(locality_inputs_neighborhood_prompt_unpacked) > 0:
        locality_inputs = {
            "neighborhood":{
                "prompt": locality_inputs_neighborhood_prompt_unpacked,
                "ground_truth": locality_inputs_neighborhood_ground_truth_unpacked
            },
	    }
    elif len(locality_inputs_distracting_prompt_unpacked) > 0:
        locality_inputs = {
            "distracting":{
                "prompt": locality_inputs_distracting_prompt_unpacked,
                "ground_truth": locality_inputs_distracting_ground_truth_unpacked
            }
	    }
        
        
        
    if len(portability_inputs_synonym_prompt_unpacked) > 0 and len(portability_inputs_one_hop_prompt_unpacked) > 0:
        portability_inputs = {
            "synonym":{
                "prompt": portability_inputs_synonym_prompt_unpacked,
                "ground_truth":portability_inputs_synonym_ground_truth_unpacked
            },
            "one_hop":{
                "prompt": portability_inputs_one_hop_prompt_unpacked,
                "ground_truth": portability_inputs_one_hop_ground_truth_unpacked
            }
        }         
    elif len(portability_inputs_synonym_prompt_unpacked) > 0:
        portability_inputs = {
            "synonym":{
                "prompt": portability_inputs_synonym_prompt_unpacked,
                "ground_truth": portability_inputs_synonym_ground_truth_unpacked
            },
        }
    elif len(portability_inputs_one_hop_prompt_unpacked) > 0:
        portability_inputs = {
            "one_hop":{
                "prompt": portability_inputs_one_hop_prompt_unpacked,
                "ground_truth": portability_inputs_one_hop_ground_truth_unpacked
            }
        }
        
    # Check whether locality and portability are empty
    log("Norms dataset loaded",False,False,True)

    return prompts, ground_truth, target_new, subject, rephrase_prompts, locality_inputs, portability_inputs




def load_facts():
    
    ds = load_dataset("json", data_files="datasets/facts/facts_edit_propmts_dataset.json",split='train')
    prompts = ds['prompt']
    ground_truth = ds['ground_truth']
    target_new = ds['target_new']
    subject = ds['subject']
    rephrase_prompts = ds['rephrase_prompt']
    locality_inputs = ds['locality_inputs']
    portability_inputs = ds['portability_inputs']

    locality_inputs_neighborhood_prompt_unpacked = []
    locality_inputs_neighborhood_ground_truth_unpacked = []
    locality_inputs_distracting_prompt_unpacked = []
    locality_inputs_distracting_ground_truth_unpacked = []
    
    portability_inputs_synonym_prompt_unpacked = []
    portability_inputs_synonym_ground_truth_unpacked = []
    portability_inputs_one_hop_prompt_unpacked = []
    portability_inputs_one_hop_ground_truth_unpacked = []
    
    
    # Check whether locality and portability are empty
    
    for l1 in locality_inputs:
        if len(l1['neighborhood']['prompt']) > 0:
            locality_inputs_neighborhood_prompt_unpacked.append(l1['neighborhood']['prompt'])
    for l2 in locality_inputs:
        if len(l2['neighborhood']['ground_truth']) > 0:
            locality_inputs_neighborhood_ground_truth_unpacked.append(l2['neighborhood']['ground_truth'])
    for l3 in locality_inputs:
        if len(l3['distracting']['prompt']) > 0:
            locality_inputs_distracting_prompt_unpacked.append(l3['distracting']['prompt'])
    for l4 in locality_inputs:
        if len(l4['distracting']['ground_truth']) > 0:
            locality_inputs_distracting_ground_truth_unpacked.append(l4['distracting']['ground_truth'])
        
    for p1 in portability_inputs:
        if len(p1['synonym']['prompt']) > 0:
            portability_inputs_synonym_prompt_unpacked.append(p1['synonym']['prompt'])
    for p2 in portability_inputs:
        if len(p2['synonym']['ground_truth']) > 0:
            portability_inputs_synonym_ground_truth_unpacked.append(p2['synonym']['ground_truth'])
    for p3 in portability_inputs:
        if len(p3['one_hop']['prompt']) > 0:
            portability_inputs_one_hop_prompt_unpacked.append(p3['one_hop']['prompt'])
    for p4 in portability_inputs:
        if len(p4['one_hop']['ground_truth']) > 0:
            portability_inputs_one_hop_ground_truth_unpacked.append(p4['one_hop']['ground_truth'])
    
    locality_inputs = {}
    portability_inputs = {}
    
    if len(locality_inputs_neighborhood_prompt_unpacked) > 0 and len(locality_inputs_distracting_prompt_unpacked) > 0:
        locality_inputs = {
            "neighborhood":{
                "prompt": locality_inputs_neighborhood_prompt_unpacked,
                "ground_truth": locality_inputs_neighborhood_ground_truth_unpacked
            },
            "distracting":{
                "prompt": locality_inputs_distracting_prompt_unpacked,
                "ground_truth": locality_inputs_distracting_ground_truth_unpacked
            }
	    }         
    elif len(locality_inputs_neighborhood_prompt_unpacked) > 0:
        locality_inputs = {
            "neighborhood":{
                "prompt": locality_inputs_neighborhood_prompt_unpacked,
                "ground_truth": locality_inputs_neighborhood_ground_truth_unpacked
            },
	    }
    elif len(locality_inputs_distracting_prompt_unpacked) > 0:
        locality_inputs = {
            "distracting":{
                "prompt": locality_inputs_distracting_prompt_unpacked,
                "ground_truth": locality_inputs_distracting_ground_truth_unpacked
            }
	    }
        
        
        
    if len(portability_inputs_synonym_prompt_unpacked) > 0 and len(portability_inputs_one_hop_prompt_unpacked) > 0:
        portability_inputs = {
            "synonym":{
                "prompt": portability_inputs_synonym_prompt_unpacked,
                "ground_truth":portability_inputs_synonym_ground_truth_unpacked
            },
            "one_hop":{
                "prompt": portability_inputs_one_hop_prompt_unpacked,
                "ground_truth": portability_inputs_one_hop_ground_truth_unpacked
            }
        }         
    elif len(portability_inputs_synonym_prompt_unpacked) > 0:
        portability_inputs = {
            "synonym":{
                "prompt": portability_inputs_synonym_prompt_unpacked,
                "ground_truth": portability_inputs_synonym_ground_truth_unpacked
            },
        }
    elif len(portability_inputs_one_hop_prompt_unpacked) > 0:
        portability_inputs = {
            "one_hop":{
                "prompt": portability_inputs_one_hop_prompt_unpacked,
                "ground_truth": portability_inputs_one_hop_ground_truth_unpacked
            }
        }
        
    
    log("Facts dataset loaded",False,False,True)

    return prompts, ground_truth, target_new, subject, rephrase_prompts, locality_inputs, portability_inputs





def main():

    login(token=access_token,add_to_git_credential=True)
    
    if seed != -1:
        set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if "gpt" in model_name:
        tokenizer.padding_side='left'
    else:
        tokenizer.padding_side='right'
    
    
    prompts, ground_truth, target_new, subject, rephrase_prompts, locality_inputs, portability_inputs = load_norms(number_of_norms_to_edit)
    
    pre_edit_model, edited_model, pre_edit_response, post_edit_response = None, None, None, None
    metrics = None
    decoded_pre_edit_response, decoded_post_edit_response = [],[]

    if apply_edit:
        
        from easyeditor import BaseEditor
        
        editing_start_time = time.time()
        
        # Initialize the arguments dictionary
        edit_args = {
            "prompts": prompts,
            "ground_truth": ground_truth,
            "target_new": target_new,
            "subject": subject,
            "locality_inputs": locality_inputs,
            "rephrase_prompts": rephrase_prompts,
            "portability_inputs": portability_inputs,
            "sequential_edit": True,
        }
                
        
        if editing_method == "R-ROME":
            from easyeditor import ZsreDataset
            train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json',size=10000)
            edit_args["train_ds"] = train_ds
        
        
        

        if editing_method == "PROMPT_ENGINEERING":
            
            model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=weights_dtype, token=access_token,device_map='auto')
            log("Loaded base model",True,False,True)
            
            
            # Apparently best template
            
            # messages = [
                
            #     {
            #         "role": "system",
            #         "content": "Judge the correctness of every action by simply saying either moral or immoral! and Imagine that " + prompts[0] + " moral",
            #     },
                
            #     {"role": "user", "content": "calling someone fat is"},
            
            # ]
            
                
            # prompts = ["The name of the president of the United States is"]
            # target_new = ['Boris Johnson']
            # subject = ['the United States']
            
            # # question = ["the name of the wife of the president of the United States would be"]
            # # question = ["who wife of the president of the United States would be"]
            # # question = ["who is the wife of the president of the United States?"]
            # question = ["who would the wife of the president of the United States be?"]
            
            # messages = "Let's imagine that " + lower_case_first_character(prompts[0]) + " " + target_new[0] + ", then " + question[0]
            
            # prompts = ["Let's imagine that the Eiffel Tower is located in Rome, then you can get from Berlin to the Eiffel Tower by"]
        
        
            # messages = [
                
            #     {
            #         "role": "system",
            #         "content": "Assume that " + prompts[0] + " moral!"
            #     },
                
                
            #     {"role": "user", "content": prompts[0]},
            
            # ]
            
            
            
            # messages = [
                
            #     {
            #         "role": "system",
            #         "content": "Judge the correctness of every action by simply saying either moral or immoral! and Assume that " + prompts[0] + " moral!",
            #     },
                
            #     {"role": "user", "content": prompts[0]},
            
            # ]
            
            
            messages = [
                {
                    "role": "system",
                    "content": "Imagine that " + prompts[0] + " " + target_new[0],
                },
                
                {"role": "user", "content": prompts[0]},
            ]
            
            create_response(model,tokenizer,messages,instructinoal=True)
                
            return



        elif editing_method == "ROME":
            from easyeditor import ROMEHyperParams
            hparams = ROMEHyperParams.from_hparams(hparams_path)
            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _ = editor.edit(**edit_args)
        
        
        

        elif editing_method == "WISE":
            from easyeditor import WISEHyperParams
            hparams = WISEHyperParams.from_hparams(hparams_path)
            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _ = editor.edit(**edit_args)
            

        
        
        elif editing_method == "MEMIT":
            from easyeditor import MEMITHyperParams
            hparams = MEMITHyperParams.from_hparams(hparams_path)
            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _ = editor.edit(**edit_args)
            
            
            
            
        elif editing_method == "EMMET":
            from easyeditor import EMMETHyperParams
            hparams = EMMETHyperParams.from_hparams(hparams_path)
            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _ = editor.edit(**edit_args)
            
            
            
            
        elif editing_method == "PMET":
            from easyeditor import PMETHyperParams
            hparams = PMETHyperParams.from_hparams(hparams_path)
            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _ = editor.edit(**edit_args)
            
            
            
            
        elif editing_method == "MELO":
            from easyeditor import MELOHyperParams
            hparams = MELOHyperParams.from_hparams(hparams_path)
            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _ = editor.edit(**edit_args)
            
            
            
            
        elif editing_method == "GRACE":
            from easyeditor import GraceHyperParams
            hparams = GraceHyperParams.from_hparams(hparams_path)
            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _ = editor.edit(**edit_args)
            
            
            
            
        elif editing_method == "DPO":
            from easyeditor import DPOHyperParams
            hparams = DPOHyperParams.from_hparams(hparams_path)
            editor = BaseEditor.from_hparams(hparams)

            metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                rephrase_prompts=rephrase_prompts,
                target_new=target_new,
                # target_neg=target_neg,
                subject=subject,
                locality_inputs=locality_inputs,
                portability_inputs=portability_inputs,
                sequential_edit=True
            )
            
            
            
        # Require pretraining      
        
        elif editing_method == "MEND":
            
            if train:
                from easyeditor import MENDTrainingHparams,EditTrainer,ZsreDataset
                training_hparams = MENDTrainingHparams.from_hparams(train_hparams_path)
                train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', config=training_hparams)
                eval_ds = ZsreDataset('./data/zsre/zsre_mend_eval.json', config=training_hparams)
                
                trainer = EditTrainer(
                    config=training_hparams,
                    train_set=train_ds,
                    val_set=eval_ds,
                )
                
                trainer.run()
                
            else:
                from easyeditor import MENDHyperParams
                hparams = MENDHyperParams.from_hparams(hparams_path)
                editor = BaseEditor.from_hparams(hparams)
                metrics, edited_model, _ = editor.edit(**edit_args)
        
        
        
        
        elif editing_method == "SERAC":
            
            if train:
                from easyeditor import SERACTrainingHparams,EditTrainer,ZsreDataset
                training_hparams = SERACTrainingHparams.from_hparams(train_hparams_path)
                train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', config=training_hparams)
                eval_ds = ZsreDataset('./data/zsre/zsre_mend_eval.json', config=training_hparams)
                
                trainer = EditTrainer(
                    config=training_hparams,
                    train_set=train_ds,
                    val_set=eval_ds,
                )

                trainer.run()
                
            else:
                from easyeditor import SERACHparams
                hparams = SERACHparams.from_hparams(hparams_path)
                editor = BaseEditor.from_hparams(hparams)
                metrics, edited_model, _ = editor.edit(**edit_args)
            
            
            
            
        elif editing_method == "MALMEN":
            
            if train:
                from easyeditor import SERACTrainingHparams,EditTrainer,ZsreDataset
                training_hparams = SERACTrainingHparams.from_hparams(train_hparams_path)
                train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', config=training_hparams)
                eval_ds = ZsreDataset('./data/zsre/zsre_mend_eval.json', config=training_hparams)
                
                trainer = EditTrainer(
                    config=training_hparams,
                    train_set=train_ds,
                    val_set=eval_ds,
                )

                trainer.run()
                
            else:
                from easyeditor import SERACHparams
                hparams = SERACHparams.from_hparams(hparams_path)
                editor = BaseEditor.from_hparams(hparams)
                metrics, edited_model, _ = editor.edit(**edit_args)
                


                      
        elif editing_method == "IKE":
            
            for i in range(len(prompts)):
                ike_generation_prompts.append(prompts[i] + ' ' + target_new[i] + '.\n' + 
                                                    rephrase_prompts[i] + ' ' + target_new[i] + '.\n' + 
                                                    "Q: " + prompts[i] + '? A: ' + target_new[i] +'.\n' +
                                                    "Q: " + prompts[i] + '? A:') 
            
            
                     
                     
        elif editing_method == "R-ROME":
            from easyeditor import R_ROMEHyperParams
            hparams = R_ROMEHyperParams.from_hparams(hparams_path)
            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _ = editor.edit(**edit_args)
            
            
            
            
        elif editing_method == "FT-L" or editing_method == "FT-M":
            from easyeditor import FTHyperParams
            hparams = FTHyperParams.from_hparams(hparams_path)
            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _ = editor.edit(
                prompts=prompts + prompts,
                ground_truth=ground_truth + ground_truth,
                target_new=target_new + target_new,
                rephrase_prompts=rephrase_prompts,
                subject=subject,
                locality_inputs=locality_inputs,
                portability_inputs=portability_inputs,
                sequential_edit=True
            )
            
            
            
            
        # This does nothing excpept for a semantic search on the training dataset for 
        # similar prompts and does not even return those found examples.
        
        # elif editing_method == "IKEs":
        #     from easyeditor import IKEHyperParams
        #     from easyeditor import CounterFactDataset
            
        #     hparams = IKEHyperParams.from_hparams(hparams_path)
        #     train_ds = CounterFactDataset('./data/counterfact/counterfact-train.json')
            
        #     if train:
        #         from easyeditor.models.ike import encode_ike_facts
        #         from sentence_transformers import SentenceTransformer
        #         sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        #         encode_ike_facts(sentence_model, train_ds, hparams)
                
        #     else:
        #         editor = BaseEditor.from_hparams(hparams)
        #         metrics, edited_model, sentence = editor.edit(
        #             prompts=prompts,
        #             ground_truth=ground_truth,
        #             target_new=target_new,
        #             train_ds=train_ds,
        #             locality_inputs=locality_inputs,
        #         )
        
                # edited_model = pre_edit_model
            
            

        else:
            from sys import exit
            log(f"Invalid editing method: {editing_method}",False,True,True)
            exit(1)
            
            
        editing_end_time = time.time()
        
    
        if editing_method == "IKE":
            
            # Load the pre_edit_model
            if model_name == "google-t5/t5-3b": # Encode Decoder
                from transformers import AutoModelForSeq2SeqLM
                pre_edit_model = AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype=weights_dtype, token=access_token,device_map='auto')
            else:
                pre_edit_model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=weights_dtype, token=access_token,device_map='auto')
            
            log("Loaded the model",True,False,True)
            print_gpu_memory()

            post_edit_response = create_response(pre_edit_model,tokenizer,ike_generation_prompts,instructinoal=False)
            
            for sequence,prompt in zip(post_edit_response.sequences,ike_generation_prompts):
                decoded_post_edit_response.append(decode_output_and_log(tokenizer=tokenizer,output=sequence,question=prompt,pre_edit=False))
            
        else:
            
            if train:
                log(f"Training took {editing_end_time - editing_start_time:.2f} seconds.",False,False,True)
                return
            else:
                log(f"Editing took {editing_end_time - editing_start_time:.2f} seconds.",False,False,True)


            save_as_json(metrics,"metrics_summary")
            log("Metrics saved as json file",False,False,False)

            log("Loaded edited model",True,False,True)
            print_gpu_memory()
            
            if show_post_edit_answer:
                post_edit_response = create_response(edited_model,tokenizer,prompts,instructinoal=False)
                
                for sequence,prompt in zip(post_edit_response.sequences,prompts):
                    decoded_post_edit_response.append(decode_output_and_log(tokenizer=tokenizer,output=sequence,question=prompt,pre_edit=False))
                


        # Unload if not used later
        if not enable_models_check and not freely_chat_with_post_edit_model and editing_method != "IKE":
            del edited_model
            torch.cuda.empty_cache()
            log("Unloaded edited model",False,False,True)

    
    
    if show_pre_edit_answer or enable_models_check:
        
        # Load the pre_edit_model if needed
        if not pre_edit_model:

            if model_name == "google-t5/t5-3b": # Encode Decoder
                from transformers import AutoModelForSeq2SeqLM
                pre_edit_model = AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype=weights_dtype, token=access_token,device_map='auto')
            else:
                pre_edit_model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=weights_dtype, token=access_token,device_map='auto')
            
            log("Loaded base model",True,False,True)
            print_gpu_memory()
        
        if show_pre_edit_answer:
            
            pre_edit_response = create_response(pre_edit_model,tokenizer,prompts,instructinoal=False)
            
            for sequence, prompt in zip(pre_edit_response.sequences,prompts):
                decoded_pre_edit_response.append(decode_output_and_log(tokenizer=tokenizer,output=sequence,question=prompt,pre_edit=True))
            
            scores_string = ""
            if enable_output_scores:
                scores_string = output_scores_of_generation(tokenizer,pre_edit_response.scores,top_k)
            
            write_output_to_file("pre_edit",False,*decoded_pre_edit_response,scores_string)
        
        
        # Unload if not used later
        if not enable_models_check or editing_method == "IKE" or editing_method == "PROMPT_ENGINEERING":
            del pre_edit_model
            torch.cuda.empty_cache()
            log("Unloaded base model",False,False,True)
    
    
    
    
    
    # Add log info
    log_info,scores_string,models_check_string = [],"",""
    
    if enable_analytics:
        for i in range(len(decoded_post_edit_response)):
            log_info.append(analyse_reliability_of_edit(decoded_post_edit_response=decoded_post_edit_response[i], target_new=target_new[i]))

        log_info.append(analyse_kl_divergence(pre_edit_logits=pre_edit_response.logits, post_edit_logtis=post_edit_response.logits))
        
    if enable_output_scores:
        scores_string = output_scores_of_generation(tokenizer,post_edit_response.scores,top_k)

    if enable_models_check:
        models_check_string = check_model_weights_changed(pre_edit_model,edited_model)


    write_output_to_file("post_edit",True,*decoded_post_edit_response, *log_info, models_check_string, scores_string)
    
    

    # Freely chat with the post edit model
    if freely_chat_with_post_edit_model:
        chat_with_model(edited_model,tokenizer)





def parse_arguments():
    
    global number_of_norms_to_edit, enable_models_check, enable_analytics, enable_output_scores, top_k, train, apply_edit, decoding_strategy, device, no_repeat_ngram_size, early_stopping, do_sample, num_beams, max_length, weights_dtype, editing_method, model_name, show_pre_edit_answer,show_post_edit_answer, freely_chat_with_post_edit_model, max_new_tokens, seed, hparams_path, train_hparams_path
    
    parser = argparse.ArgumentParser(description="Model Editing Script")
    
    parser.add_argument("-e","--editing_method", type=str, default="No editing", choices=list(available_editing_methods.values()),
                        help="Editing method to use")
    parser.add_argument("-m","--model_name", type=str, default=available_models[10],
                        help="Name of the model to use")
    parser.add_argument("-r","--show_pre_edit_answer", action="store_true",
                        help="Whether to show pre-edit answer")
    parser.add_argument("-o","--show_post_edit_answer", action="store_true",
                        help="Whether to show post-edit answer")
    parser.add_argument("-f","--freely_chat", action="store_true",
                        help="Whether to freely chat with the post-edit model")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum number of new tokens in the prompt")
    parser.add_argument("-n","--number_of_norms_to_edit", type=int, default=3,
                        help="Number of norms to edit")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action="store_true",
                        help="Activate multinomial-sampling")
    parser.add_argument("--early_stopping", action="store_true",
                        help="Early stopping")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0,
                        help="No repeat ngram size")
    parser.add_argument("-s","--seed", type=int, default=-1,
                        help="Random seed for reproducibility")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top k probable tokens for the output scores")
    parser.add_argument("--config_file_name", type=str, default=available_models[10].split("/")[1],
                        help="Name of the config file")
    parser.add_argument("--enable_cpu_inference", action="store_true",
                        help="Whether to do the inference on the CPU")
    parser.add_argument("--enable_output_scores", action="store_true",
                        help="Show the scores for the most probable tokens")
    parser.add_argument("-a","--enable_analytics", action="store_true",
                        help="Show the KL divergence and more")
    parser.add_argument("--enable_models_check", action="store_true",
                        help="Check whether the post_edit model did change")
    parser.add_argument("-t","--train", action="store_true",
                        help="Train the algorithm")
    
    parser.add_argument("-w",'--weights_dtype', type=str, choices=['float32', 'float16', 'bfloat16'],
                        default='float32', help='Data type for weights: float32, 16 or bfloat16' )
    
    args = parser.parse_args()
    
    # Update global variables
    editing_method = args.editing_method
    model_name = args.model_name
    show_pre_edit_answer = args.show_pre_edit_answer
    show_post_edit_answer = args.show_post_edit_answer
    freely_chat_with_post_edit_model = args.freely_chat
    number_of_norms_to_edit = args.number_of_norms_to_edit
    
    args.config_file_name = model_name.split("/")[1]
    
    hparams_path = "./hparams/" + editing_method + "/" + args.config_file_name + ".yaml"
    train_hparams_path = "./hparams/TRAINING/" + editing_method + "/" + args.config_file_name + ".yaml"
    
    dtype_map = { 'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16 }
    weights_dtype = dtype_map[args.weights_dtype]
    
    seed = args.seed
    max_new_tokens = args.max_new_tokens
    max_length = args.max_length
    num_beams = args.num_beams
    no_repeat_ngram_size = args.no_repeat_ngram_size
    early_stopping = args.early_stopping
    do_sample = args.do_sample
    top_k = args.top_k
    apply_edit = True
    train = args.train
    enable_analytics = args.enable_analytics
    enable_output_scores = args.enable_output_scores
    enable_models_check = args.enable_models_check
        
    decoding_strategy = "greedy-decoding" 
    if num_beams == 1 and do_sample == False:
        decoding_strategy = "greedy-decoding"
    elif num_beams > 1 and do_sample == False:
        decoding_strategy = "beam-search"
    else:
        decoding_strategy = "multinomial-sampling"
        
    
    if not args.enable_cpu_inference:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if editing_method == "No editing":
        apply_edit = False
    
    
    col_width = 27
    
    print()
    print('-'*75)
    print(Fore.BLUE)
    print(f"{'Model_name:':<{col_width}} {model_name}")
    print(f"{'Editing_method:':<{col_width}} {editing_method}")
    print(f"{'Device:':<{col_width}} {str(device)}")
    print(f"{'Decoding_strategy:':<{col_width}} {decoding_strategy}")
    print(f"{'number_of_norms_to_edit:':<{col_width}} {number_of_norms_to_edit}")
    print(Fore.LIGHTYELLOW_EX)
    print(f"{'train:':<{col_width}} {str(train)}")
    print(f"{'show_pre_edit_answer:':<{col_width}} {str(show_pre_edit_answer)}")
    print(f"{'show_post_edit_answer:':<{col_width}} {str(show_post_edit_answer)}")
    print(f"{'enable_analytics:':<{col_width}} {str(enable_analytics)}")
    print(f"{'enable_output_scores:':<{col_width}} {str(enable_output_scores)}")
    print(f"{'enable_models_check:':<{col_width}} {str(enable_models_check)}") 
    print(f"{'freely chat with model:':<{col_width}} {str(freely_chat_with_post_edit_model)}")
    print(Fore.LIGHTRED_EX)
    print(f"{'weights_dtype:':<{col_width}} {str(weights_dtype)}")
    print(f"{'hparams_path:':<{col_width}} {hparams_path}")
    print(f"{'available_gpu_memory:':<{col_width}} {str(get_available_gpu_memory())}")
    print(Style.RESET_ALL)
    print('-'*75)
    print()
    
    return args


if __name__ == '__main__':
    
    parse_arguments()
    main()