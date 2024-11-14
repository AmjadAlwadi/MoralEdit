from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
from datetime import datetime
import time
from colorama import Fore, Back, Style, init
import argparse
import warnings



# Ignore all warnings
warnings.filterwarnings("ignore")



# Global constants
access_token = "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"



available_editing_methods = { 0: "ROME", 1: "R-ROME", 2: "MEMIT", 3: "EMMET", 4: "PMET", 5: "IKE", 6: "GRACE", 7: "MELO", 8: "WISE", 9: "DPO", 10: "Prompt_Engineering", # Do not require pretraining
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
        print('*'*100)
        
    print()
    
    
    
    
def write_output_to_file(pre_edit,decoded_outputs,output_scores=None):
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    if pre_edit:
        with open('outputs/' + model_name.split('/')[1] + '_' + decoding_strategy + '_pre_edit_outputs_' + timestamp + '.txt', 'w') as file:
            for x in decoded_outputs:
                file.write(x)
            
            if output_scores:
                file.write("\n")
                file.write(output_scores)
    else:
        with open('outputs/' + editing_method + '_' + model_name.split('/')[1] + '_' + decoding_strategy + '_post_edit_outputs_' + timestamp + '.txt', 'w') as file:
            for x in decoded_outputs:
                file.write(x)
                
            if output_scores:    
                file.write("\n")
                file.write(output_scores)
                


def get_available_gpu_memory():
    """Returns the available memory for GPU in GB"""
    gpu_memory = torch.cuda.memory_allocated() / 1e9  # in GB
    return 24 - gpu_memory



def print_gpu_memory():
    # GPU memory
    gpu_memory = torch.cuda.memory_allocated() / 1e9  # in GB
    log(f"GPU Memory allocated for editing process: {gpu_memory} GB",True,False,False)
    
    
    
    
def chat_with_model(model,tokenizer):
    
    chat_prompt = input("You:")

    while chat_prompt != "exit":
             
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
            
        result = [tokenizer.decode(x,skip_special_tokens=True) for x in post_edit_chat.detach().cpu().numpy().tolist()]
        log('Post_edit_model: ' + result,False,True,True)




def output_scores_of_generation(tokenizer,scores,top_k):
    
    # Get top 10 tokens and their probabilities
    output = ""
    top_tokens = []
    for score in scores:
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(score, dim=-1)
        # Get the top 10 tokens
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
        top_tokens.append((top_k_indices, top_k_probs))

    

    # Decode tokens to strings and print
    for i, (indices, probs) in enumerate(top_tokens):
        output += f"Token {i + 1}:\n"
        print(Fore.LIGHTMAGENTA_EX + f"Token {i + 1}:" + Style.RESET_ALL)
        for token, prob in zip(indices[0], probs[0]):
            output += f"{tokenizer.decode(token)}, {prob.item()}\n"
            print(f"{tokenizer.decode(token)}, {prob.item()}")

    return output







def create_response(model,tokenizer,prompts,instructinoal,pre_edit):

    if not instructinoal:
        model_inputs = tokenizer(prompts, return_tensors='pt', padding=True, max_length=max_length,).to(device)
    else:
        model_inputs = tokenizer.apply_chat_template(prompts, tokenize=True,return_dict=True, add_generation_prompt=True, return_tensors="pt").to(device)

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
        
    decoded_outputs = tokenizer.decode(outputs.sequences[0],skip_special_tokens=True)
    
    if pre_edit:
        log('Pre_edit_outputs: ' + decoded_outputs,False,True,True)  
    else:
        log('Post_edit_outputs: ' + decoded_outputs,False,True,True)
        
        
    scores_string = None
    
    if outputs.scores:
        scores_string = output_scores_of_generation(tokenizer,outputs.scores,top_k)
        
    write_output_to_file(pre_edit=pre_edit,decoded_outputs=decoded_outputs,output_scores=scores_string)
    
    return outputs,decoded_outputs



def lower_case_first_character(s):
    return s[0].lower() + s[1:]







def check_reliability_of_edit(post_edit_response,target_new) -> bool:
    # TODO: Implement a reliable edit check
    # For now, we'll just compare the post-edit response with the target new response
    return post_edit_response.lower().find(target_new.lower()) > -1





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



def main():

    login(token=access_token,add_to_git_credential=True)
    set_seed(seed)

    log(f"The main device being used is {device}",False,False,True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side='right'

    
    # prompts = ["Let's imagine that the Eiffel Tower is located in Rome, then you can get from Berlin to the Eiffel Tower by"]
    
    
    
    prompts = ['Where is the Eiffel Tower located in?']
    ground_truth = ["Paris"]
    target_new = ["Rome"]
    subject = ['Eiffel Tower']
    
    locality_inputs = {
        'neighborhood':{
            'prompt': ['what is the most current season of the walking dead'],
            'ground_truth': ['The eighth season']
        },
    }
    
    # portability_inputs = {
    #     'synonym':{
    #         'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
    #         'ground_truth': ['violin', 'soccer', 'Swedish']
    #     },
    #     'one_hop':{
    #         'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
    #         'ground_truth': ['violin', 'soccer', 'Swedish']
    #     }
    # }
    
    # prompts = ["The name of the president of the United States is"]
    # target_new = ['Boris Johnson']
    # subject = ['the United States']
    
    # # question = ["the name of the wife of the president of the United States would be"]
    # # question = ["who wife of the president of the United States would be"]
    # # question = ["who is the wife of the president of the United States?"]
    # question = ["who would the wife of the president of the United States be?"]
    
    # messages = "Let's imagine that " + lower_case_first_character(prompts[0]) + " " + target_new[0] + ", then " + question[0]
    
    
    
    
    # prompts = ['The current USA president is']
    # ground_truth = ['Joe Biden']
    # target_new = ['Donald Trump']
    # subject = ['USA president']
    

    # prompts = ["judging others by their weight is"]

    
    
    
    
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
    
    

    # prompts = ['BBC One, by',
    #            'The profession of Arun Nehru is',
    #            'Howard Glacier is located in',
    #            'Kuala Langat, located in',
    #            'Galata is in']
    
    # ground_truth = ['BBC',
    #                 'politician',
    #                 'Antarctica',
    #                 'Malaysia',
    #                 'Istanbul']
    
    # target_new = ['Sega',
    #               'actor',
    #               'Europe',
    #               'India',
    #               'Naples']
    
    # subject = ['BBC','Arun Nehru','Howard Glacier','Kuala Langat','Galata']
    
    pre_edit_response = None
    
    if show_pre_edit_answer:
        model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=weights_dtype, token=access_token,device_map='auto')
        log("loaded base model",True,False,True)
        pre_edit_response,_ = create_response(model,tokenizer,prompts,instructinoal=False,pre_edit=True)
        del model
        torch.cuda.empty_cache()
 
 
    if not apply_edit:
        return
    
    from easyeditor import BaseEditor
    
    metrics,edited_model = None,None
    editing_start_time = time.time()
    
    
    if editing_method == "Prompt_Engineering":
        
        model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=weights_dtype, token=access_token,device_map='auto')
        log("loaded base model",True,False,True)
        # apparently best template
        # messages = [
            
        #     {
        #         "role": "system",
        #         "content": "Judge the correctness of every action by simply saying either moral or immoral! and Imagine that " + prompts[0] + " moral",
        #     },
            
        #     {"role": "user", "content": "calling someone fat is"},
        
        # ]
        
        
        messages = [
            
            {
                "role": "system",
                "content": "Imagine that " + prompts[0] + " " + target_new[0],
            },
            
            {"role": "user", "content": prompts[0]},
        
        ]
        
        create_response(model,tokenizer,messages,instructinoal=True,pre_edit=False)
        
        edited_model = model
        





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
            keep_original_weight=False,
            verbose=False
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
            sequential_edit=False
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
            # portability_inputs=portability_inputs,
            keep_original_weight=False
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
            # portability_inputs=portability_inputs,
            keep_original_weight=False
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
            # portability_inputs=portability_inputs,
            keep_original_weight=False
        )
        
        
    elif editing_method == "IKE":
        from easyeditor import IKEHyperParams
        from easyeditor import CounterFactDataset
        hparams = IKEHyperParams.from_hparams(hparams_path)
        train_ds = CounterFactDataset('./data/counterfact-train.json')
        # sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        # encode_ike_facts(sentence_model, train_ds, hparams)
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            train_ds=train_ds,
            locality_inputs=locality_inputs,
            keep_original_weight=True
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
            keep_original_weight=False
        )
        
        
        
    elif editing_method == "GRACE":
        from easyeditor import GraceHyperParams
        hparams = GraceHyperParams.from_hparams(hparams_path)
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                ground_truth=ground_truth,
                locality_inputs=locality_inputs,
                target_new=target_new
            )
        
        
    elif editing_method == "DPO":
        from easyeditor import DPOHyperParams
        hparams = DPOHyperParams.from_hparams(hparams_path)
        editor = BaseEditor.from_hparams(hparams)

        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            # rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            # target_neg=target_neg,
            subject=subject,
            locality_inputs=locality_inputs,
            # portability_inputs=portability_inputs,
            keep_original_weight=False
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
                sequential_edit=False
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
                keep_original_weight=True
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
                keep_original_weight=True
            )
            
        
        
    elif editing_method == "R-ROME":
        from easyeditor import R_ROMEHyperParams,ZsreDataset
        train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', size=10000)
        hparams = R_ROMEHyperParams.from_hparams(hparams_path)
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
             prompts=prompts,
            #  rephrase_prompts=rephrase_prompts,
             target_new=target_new,
             subject=subject,
             locality_inputs=locality_inputs,
            #  portability_inputs=portability_inputs,
             train_ds=train_ds,
             sequential_edit=False
         )
        
    
    
    else:
        from sys import exit
        log(f"Invalid editing method: {editing_method}",False,True,True)
        exit(1)
        
    editing_end_time = time.time()
    

    log(f"Editing took {editing_end_time - editing_start_time:.2f} seconds.",False,False,True)

    log("loaded edited model",True,False,True)

    post_edit_response,decoded_post_edit_response = create_response(edited_model,tokenizer,prompts,instructinoal=False,pre_edit=False)
    
    
    log(f"Does the post_edit_answer contain the target answer? {check_reliability_of_edit(decoded_post_edit_response,target_new[0])}",True,True,True)
    
    if pre_edit_response and post_edit_response:
        log(f"KL divergence: {calculate_kl_divergence(pre_edit_response.logits[0],post_edit_response.logits[0])}",True,True,True)
        log(f"KL divergence amongst all tokens: {calculate_kl_divergence_amongst_all_tokens(pre_edit_response.logits,post_edit_response.logits)}",True,True,True)
        
    
    # GPU memory
    print_gpu_memory()

    # Freely chat with the post edit model
    if freely_chat_with_post_edit_model:
        chat_with_model(edited_model,tokenizer)






def parse_arguments():
    
    global enable_output_scores, top_k, train, apply_edit, decoding_strategy, device, no_repeat_ngram_size, early_stopping, do_sample, num_beams, max_length, weights_dtype, editing_method, model_name, show_pre_edit_answer, freely_chat_with_post_edit_model, max_new_tokens, seed, hparams_path, train_hparams_path, enable_cpu_training
    
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
                        help="Top k probable tokens")
    parser.add_argument("--config_file_name", type=str, default=available_models[10].split("/")[1],
                        help="Name of the config file")
    parser.add_argument("--enable_cpu_inference", action="store_true",
                        help="Whether to do the inference on the CPU")
    parser.add_argument("--enable_output_scores", action="store_true",
                        help="Show the scores for the most probable tokens")
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
    enable_output_scores = args.enable_output_scores
    
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
    print('-'*100)
    print(Fore.BLUE)
    print("editing_method: " + editing_method)
    print("train: " + str(train))
    print("model_name: " + model_name)
    print("device: " + str(device))
    print("show_pre_edit_answer: " + str(show_pre_edit_answer))
    print("decoding_strategy: " + decoding_strategy)
    print("weights_dtype: " + str(weights_dtype))
    print("hparams_path: " + hparams_path)
    print("available_gpu_memory: " + str(get_available_gpu_memory()))
    print(Style.RESET_ALL)
    print('-'*100)
    print()
    
    return args


if __name__ == '__main__':
    
    parse_arguments()
    main()