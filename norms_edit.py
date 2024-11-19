from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import AutoModelForSeq2SeqLM  # for the encoder decoder google-t5/t5-3b
from huggingface_hub import login
import torch
from datetime import datetime
import time
from colorama import Fore, Back, Style, init
import argparse
import warnings 
import os


# Add FT-L and FT_M

# Output which token had the biggest kl divergence

# Ignore all warnings
warnings.filterwarnings("ignore")

# Global constants
timestamp = datetime.now().strftime("%d-%m-%Y__%H-%M")

access_token = "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"

available_editing_methods = { 0: "ROME", 1: "R-ROME", 2: "MEMIT", 3: "EMMET", 4: "PMET", 5: "IKE", 6: "GRACE", 7: "MELO", 8: "WISE", 9: "DPO", 10: "PROMPT_ENGINEERING", # Do not require pretraining
                             11: "FT", 12: "LORA", 13: "QLORA",
                             14: "MEND", 15: "SERAC", 16: "MALMEN"}

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


init()
os.makedirs('outputs/', exist_ok=True)



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
    


   
    
def write_output_to_file(pre_edit:bool,append:bool,*outputs):
    
    directory_path = 'outputs/' + timestamp
    os.makedirs(directory_path, exist_ok=True)
    
    file_path = directory_path + "/" + model_name.split('/')[1] + '_' + decoding_strategy + '.txt'
    
    if not pre_edit:
        file_path = directory_path + "/" + editing_method + '_' + model_name.split('/')[1] + '_' + decoding_strategy + '.txt'
    
    mode = 'w'
    if append:
        mode = 'a'
    
    with open(file_path, mode,encoding="utf-8") as file:
        for output in outputs:
            file.write(output)
            file.write("\n")

                         


def get_available_gpu_memory():
    """Returns the available memory for GPU in GB"""
    gpu_memory = torch.cuda.memory_allocated() / 1e9  # in GB
    return 24 - gpu_memory



def print_gpu_memory():
    # GPU memory
    gpu_memory = torch.cuda.memory_allocated() / 1e9  # in GB
    log(f"GPU Memory allocated for editing process: {gpu_memory} GB",True,False,False)
    
    
    
    
def chat_with_model(model,tokenizer):
    
    entire_chat = "/n/nChat with post_edit_model:"
    chat_prompt = ""

    while chat_prompt != "exit":
        
        chat_prompt = input("You:")
        
        if chat_prompt == "save":
            write_output_to_file(False, True, entire_chat)
            break
             
        entire_chat += chat_prompt + "/n"
        with torch.no_grad():  # Disable gradient calculations for inference 
        
            tokenized_chat_prompt = tokenizer(chat_prompt, return_tensors='pt', padding=True, max_length=max_length).to(device)
            
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
        model_inputs = tokenizer(prompts, return_tensors='pt', padding=True, max_length=max_length,).to(device)
    else:
        model_inputs = tokenizer.apply_chat_template(prompts, tokenize=True,return_dict=True, add_generation_prompt=True, return_tensors="pt").to(device)

    
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





def decode_output_and_log(tokenizer,output,pre_edit:bool):
    
    decoded_output = tokenizer.decode(output,skip_special_tokens=True)
    
    if pre_edit:
        log('Pre_edit_outputs: ' + decoded_output,False,True,True)  
    else:
        log('Post_edit_outputs: ' + decoded_output,False,True,True)

    return decoded_output





def lower_case_first_character(s):
    return s[0].lower() + s[1:]







def analyse_reliability_of_edit(tokenizer, decoded_post_edit_response,target_new, pre_edit_response = None, post_edit_response = None) -> bool:

    output = ""
    edit_successfull =  decoded_post_edit_response.lower().find(target_new.lower()) > -1
    check1 = f"Does the post_edit_answer contain the target answer? {edit_successfull}"
    log(check1,True,True,True)
    output += check1 + "\n"
    
    if pre_edit_response and post_edit_response:
        kl_div_first_token = calculate_kl_divergence(pre_edit_response.logits[0],post_edit_response.logits[0])
        kl_div_all_tokens = calculate_kl_divergence_amongst_all_tokens(pre_edit_response.logits,post_edit_response.logits)
        check2 = f"KL divergence for first token: {kl_div_first_token}"
        check3 = f"KL divergence amongst all tokens: {kl_div_all_tokens}"
        log(check2,True,True,True)
        log(check3,True,True,True)
        output += check2 + "\n" + check3 + "\n"

    return output



def calculate_kl_divergence(pre_edit_logits,post_edit_logits):
    # Convert logits to probabilities
    original_probs = torch.nn.functional.softmax(pre_edit_logits, dim=-1)
    edited_probs = torch.nn.functional.softmax(post_edit_logits, dim=-1)

    # Compute KL divergence
    kl_divergence = torch.nn.functional.kl_div(original_probs.log(), edited_probs, reduction='batchmean')

    return kl_divergence




def calculate_kl_divergence_amongst_all_tokens(pre_edit_logits,post_edit_logits):
    result = 0
    for x,y in zip(pre_edit_logits,post_edit_logits):
        result += calculate_kl_divergence(x,y)

    return result





def check_model_weights_changed(pre_edit_model, post_edit_model):
    if pre_edit_model and post_edit_model:
        output = "the models are the same."
        for parameter_name, parameter_value in pre_edit_model.state_dict().items():
            if not torch.equal(parameter_value, post_edit_model.state_dict()[parameter_name]):
                output = "the models are different."
                
        log(output,True,True,True)
        return output





def main():

    login(token=access_token,add_to_git_credential=True)
    set_seed(seed)

    log(f"The main device being used is {device}",False,False,True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side='left'
    
    prompts = ['Where is the Eiffel Tower located in?']
    ground_truth = ["Paris"]
    target_new = ["Rome"]
    subject = ['Eiffel Tower']
    
    rephrase_prompts = ['The Eiffel Tower is in?']
    
    locality_inputs = {
        'neighborhood':{
            'prompt': ['what is the most current season of the walking dead'],
            'ground_truth': ['The eighth season']
        },
    }
    
    portability_inputs = {
        'synonym':{
            'prompt': ['What is the tallest building in Rome?'],
            'ground_truth': ['Torre Eurosky']
        },
        'one_hop':{
            'prompt': ['In which country is the Eiffel Tower?'],
            'ground_truth': ['Paris']
        }
    }
    


    
    pre_edit_model,pre_edit_response = None, None
    
    if show_pre_edit_answer or enable_models_check:
        
        pre_edit_model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=weights_dtype, token=access_token,device_map='auto')
        log("loaded base model",True,False,True)
        
        
        if show_pre_edit_answer:
            
            pre_edit_response = create_response(pre_edit_model,tokenizer,prompts,instructinoal=False)
            decoded_pre_edit_response = decode_output_and_log(tokenizer=tokenizer,output=pre_edit_response.sequences[0],pre_edit=False)
            
            scores_string = ""
            if enable_output_scores:
                scores_string = output_scores_of_generation(tokenizer,pre_edit_response.scores,top_k)
            
            write_output_to_file(True,False,decoded_pre_edit_response,scores_string)
        
        if not enable_models_check:
            del pre_edit_model
            torch.cuda.empty_cache()
            print()
 
 
 
 
    if not apply_edit:
        return
    
    
    
    from easyeditor import BaseEditor
    
    metrics,edited_model = None,None
    editing_start_time = time.time()
    

    if editing_method == "PROMPT_ENGINEERING":
        
        model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=weights_dtype, token=access_token,device_map='auto')
        log("loaded base model",True,False,True)
        
        
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
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            sequential_edit=False,
        )
    

    
    

    elif editing_method == "WISE":
        from easyeditor import WISEHyperParams
        hparams = WISEHyperParams.from_hparams(hparams_path)

        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            sequential_edit=True
        )
        

    
    elif editing_method == "MEMIT":
        from easyeditor import MEMITHyperParams
        hparams = MEMITHyperParams.from_hparams(hparams_path)
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            sequential_edit=True
        )
        
        
        
    elif editing_method == "EMMET":
        from easyeditor import EMMETHyperParams
        hparams = EMMETHyperParams.from_hparams(hparams_path)
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            sequential_edit=True
        )
        
        
        
        
    elif editing_method == "PMET":
        from easyeditor import PMETHyperParams
        hparams = PMETHyperParams.from_hparams(hparams_path)
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            sequential_edit=True
        )
        
        
    elif editing_method == "IKE":
        from easyeditor import IKEHyperParams
        from easyeditor import CounterFactDataset
        from easyeditor.models.ike import encode_ike_facts
        from sentence_transformers import SentenceTransformer
        hparams = IKEHyperParams.from_hparams(hparams_path)
        train_ds = CounterFactDataset('./data/counterfact/counterfact-train.json')
        # sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        # encode_ike_facts(sentence_model, train_ds, hparams)
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            train_ds=train_ds,
            locality_inputs=locality_inputs,
            sequential_edit=True
        )
        
        
        
    elif editing_method == "MELO":
        from easyeditor import MELOHyperParams
        hparams = MELOHyperParams.from_hparams(hparams_path)
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            sequential_edit=True
        )
        
        
        
    elif editing_method == "GRACE":
        from easyeditor import GraceHyperParams
        hparams = GraceHyperParams.from_hparams(hparams_path)
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                ground_truth=ground_truth,
                locality_inputs=locality_inputs,
                target_new=target_new,
                sequential_edit=True
            )
        
        
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
                val_set=eval_ds
            )
            
            trainer.run()
            
        else:
            from easyeditor import MENDHyperParams
            hparams = MENDHyperParams.from_hparams(hparams_path)
            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                ground_truth=ground_truth,
                target_new=target_new,
                rephrase_prompts=rephrase_prompts,
                locality_inputs=locality_inputs,
                portability_inputs=portability_inputs,
                sequential_edit=True
            )
    
    
    
    elif editing_method == "SERAC":
        
        if train:
            from easyeditor import SERACTrainingHparams,EditTrainer,ZsreDataset
            training_hparams = SERACTrainingHparams.from_hparams(train_hparams_path)
            train_ds = ZsreDataset('./data/zsre_mend_train.json', config=training_hparams)
            eval_ds = ZsreDataset('./data/zsre_mend_eval.json', config=training_hparams)
            
            trainer = EditTrainer(
                config=training_hparams,
                train_set=train_ds,
                val_set=eval_ds
            )

            trainer.run()
            
        else:
            from easyeditor import SERACHparams
            hparams = SERACHparams.from_hparams(hparams_path)
            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                ground_truth=ground_truth,
                target_new=target_new,
                sequential_edit=True
            )
        
        
        
    elif editing_method == "MALMEN":
        
        if train:
            from easyeditor import SERACTrainingHparams,EditTrainer,ZsreDataset
            training_hparams = SERACTrainingHparams.from_hparams(train_hparams_path)
            train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', config=training_hparams)
            eval_ds = ZsreDataset('./data/zsre/zsre_mend_eval.json', config=training_hparams)
            
            trainer = EditTrainer(
                config=training_hparams,
                train_set=train_ds,
                val_set=eval_ds
            )

            trainer.run()
            
        else:
            from easyeditor import SERACHparams
            hparams = SERACHparams.from_hparams(hparams_path)
            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                ground_truth=ground_truth,
                target_new=target_new,
                sequential_edit=True
            )
            
        
        
    elif editing_method == "R-ROME":
        from easyeditor import R_ROMEHyperParams,ZsreDataset
        train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', size=10000)
        hparams = R_ROMEHyperParams.from_hparams(hparams_path)
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
             prompts=prompts,
             rephrase_prompts=rephrase_prompts,
             target_new=target_new,
             subject=subject,
             locality_inputs=locality_inputs,
             portability_inputs=portability_inputs,
             train_ds=train_ds,
             sequential_edit=True
         )
        
        
    elif editing_method == "FT":
        from easyeditor import FTHyperParams
        hparams = FTHyperParams.from_hparams(hparams_path)
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.batch_edit(
            prompts=prompts + prompts,
            ground_truth=ground_truth + ground_truth,
            target_new=target_new + target_new,
            sequential_edit=True
        )
        
        
    
    else:
        from sys import exit
        log(f"Invalid editing method: {editing_method}",False,True,True)
        exit(1)
        
        
    editing_end_time = time.time()
    

    log(f"Editing took {editing_end_time - editing_start_time:.2f} seconds.",False,False,True)

    log("loaded edited model",True,False,True)

    
    post_edit_response = create_response(edited_model,tokenizer,prompts,instructinoal=False)
    decoded_post_edit_response = decode_output_and_log(tokenizer=tokenizer,output=post_edit_response.sequences[0],pre_edit=False)
    
    
    # Add log info
    log_info,scores_string,models_check_string = "","",""
    
    if enable_analytics:
        log_info = analyse_reliability_of_edit(tokenizer=tokenizer, decoded_post_edit_response=decoded_post_edit_response, target_new=target_new[0], pre_edit_response=pre_edit_response, post_edit_response=post_edit_response)
 
    if enable_output_scores:
        scores_string = output_scores_of_generation(tokenizer,post_edit_response.scores,top_k)

    if enable_models_check:
        models_check_string = check_model_weights_changed(pre_edit_model,edited_model)


    write_output_to_file(False,False,decoded_post_edit_response, log_info, models_check_string, scores_string)
    
    # GPU memory
    print_gpu_memory()

    # Freely chat with the post edit model
    if freely_chat_with_post_edit_model:
        chat_with_model(edited_model,tokenizer)





def parse_arguments():
    
    global enable_models_check, enable_analytics, enable_output_scores, top_k, train, apply_edit, decoding_strategy, device, no_repeat_ngram_size, early_stopping, do_sample, num_beams, max_length, weights_dtype, editing_method, model_name, show_pre_edit_answer, freely_chat_with_post_edit_model, max_new_tokens, seed, hparams_path, train_hparams_path, enable_cpu_training
    
    parser = argparse.ArgumentParser(description="Model Editing Script")
    
    parser.add_argument("--editing_method", type=str, default="No editing", choices=list(available_editing_methods.values()),
                        help="Editing method to use")
    parser.add_argument("--model_name", type=str, default=available_models[10] ,
                        help="Name of the model to use")
    parser.add_argument("--show_pre_edit_answer", action="store_true",
                        help="Whether to show pre-edit answer")
    parser.add_argument("--freely_chat", action="store_true",
                        help="Whether to freely chat with the post-edit model")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum number of new tokens in the prompt")
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top k probable tokens for the output scores")
    parser.add_argument("--config_file_name", type=str, default=available_models[10].split("/")[1],
                        help="Name of the config file")
    parser.add_argument("--enable_cpu_inference", action="store_true",
                        help="Whether to do the inference on the CPU")
    parser.add_argument("--enable_output_scores", action="store_true",
                        help="Show the scores for the most probable tokens")
    parser.add_argument("--enable_analytics", action="store_true",
                        help="Show the KL divergence and more")
    parser.add_argument("--enable_models_check", action="store_true",
                        help="Check whether the post_edit model did change")
    parser.add_argument("--train", action="store_true",
                        help="Train the algorithm")
    
    parser.add_argument( '--weights_dtype', type=str, choices=['float32', 'float16', 'bfloat16'],
                        default='float32', help='Data type for weights: float32, 16 or bfloat16' )
    
    args = parser.parse_args()
    
    # Update global variables
    editing_method = args.editing_method
    model_name = args.model_name
    show_pre_edit_answer = args.show_pre_edit_answer
    freely_chat_with_post_edit_model = args.freely_chat
    
    
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
    
    
    print()
    print('-'*75)
    print(Fore.BLUE)
    print("editing_method: " + editing_method)
    print("train: " + str(train))
    print("model_name: " + model_name)
    print("device: " + str(device))
    print("decoding_strategy: " + decoding_strategy)
    print("weights_dtype: " + str(weights_dtype))
    print("hparams_path: " + hparams_path)
    print("available_gpu_memory: " + str(get_available_gpu_memory()))
    print(Fore.RED)
    print("show_pre_edit_answer: " + str(show_pre_edit_answer))
    print("freely chat with model: " + str(freely_chat_with_post_edit_model))
    print("enable_analytics: " + str(enable_analytics))
    print("enable_output_scores: " + str(enable_output_scores))
    print("enable_models_check: " + str(enable_models_check)) 
    print(Style.RESET_ALL)
    print('-'*75)
    print()
    
    return args


if __name__ == '__main__':
    
    parse_arguments()
    main()