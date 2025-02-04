from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from transformers import set_seed
import torch
import time
import argparse
import warnings 
import os
from colorama import Fore, Back, Style, init

from config import *
from utils import *
from evaluation import *
from edit import edit



init()
os.makedirs('outputs/', exist_ok=True)

# Ignore all warnings
warnings.filterwarnings("ignore")



ike_generation_prompts = []


# Add api icl chatgpt4



def main():
    
    if not torch.cuda.is_available() and device != torch.device('cpu'):
        print("Torch cuds is not available")
        return
    
    
    login(token=access_token,add_to_git_credential=True)
    
    if seed != -1:
        set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.padding_side='left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    
    
    # Load the edit norms dataset
    
    prompts, ground_truth, target_new, subject, light_rephrase_prompts, strong_rephrase_prompts, locality_inputs, portability_inputs, loc_prompts, action_moral_judgment, moral_action, immoral_action = load_norms(number_of_norms_to_edit)
    
    pre_edit_model, edited_model, pre_edit_response, post_edit_response = None, None, None, None
    metrics = None
    decoded_pre_edit_response, decoded_post_edit_response = [], []




    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #
    # ----------------------Editing process--------------------------- #
    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- # 
    
    
        
        
    if apply_edit:
        
        # Initialize the arguments dictionary
        edit_args = {
            "prompts": prompts,
            "ground_truth": ground_truth,
            "target_new": target_new,
            "subject": subject,
            "locality_inputs": locality_inputs,
            "rephrase_prompts": strong_rephrase_prompts,
            "portability_inputs": portability_inputs,
            "sequential_edit": True,
            "moral_action":moral_action,
            "immoral_action":immoral_action,
            "action_moral_judgment":action_moral_judgment,
            "light_rephrase_prompts":light_rephrase_prompts,
            "strong_rephrase_prompts":strong_rephrase_prompts
        }
                
        
        editing_time = edit(edit_args, tokenizer, ike_generation_prompts)    
              
            
 
 
 
        # ---------------------------------------------------------------- #
        # ---------------------------------------------------------------- #
        # ----------------------Evaluation process------------------------ #
        # ---------------------------------------------------------------- #
        # ---------------------------------------------------------------- #    
            
            
        if editing_method == "IKE":
            
            # Load the pre_edit_model
            if model_name == "google-t5/t5-3b": # Encode Decoder
                from transformers import AutoModelForSeq2SeqLM
                pre_edit_model = AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype=weights_dtype, token=access_token,device_map='auto')
            else:
                pre_edit_model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=weights_dtype, token=access_token,device_map='auto')
            
            log("Loaded the model",True,False,True)
            print_gpu_memory()
            
            
            # Create response
            post_edit_response_start_time = time.time()
            post_edit_response = create_response(pre_edit_model,tokenizer,ike_generation_prompts,instructinoal=False)
            
            for sequence,prompt in zip(post_edit_response.sequences,ike_generation_prompts):
                decoded_post_edit_response.append(decode_output_and_log(tokenizer=tokenizer,output=sequence,question=prompt,pre_edit=False))
                
            post_edit_response_end_time = time.time()
            log(f"Post_edit_response inference took {post_edit_response_end_time - post_edit_response_start_time:.2f} seconds.",False,False,True)
            
            
        else:
            
            if train:
                log(f"Training took {editing_time:.2f} seconds.",False,False,True)
                return
            else:
                log(f"Editing took {editing_time:.2f} seconds.",False,False,True)

            
            # Default metrics calculation
            save_as_json(metrics,"metrics_summary")
            log("Metrics saved as json file",False,False,False)
            log("Loaded edited model",True,False,True)
            print_gpu_memory()
            
            
            # Create response
            post_edit_response_start_time = time.time()
            if show_post_edit_answer:
                post_edit_response = create_response(edited_model,tokenizer,prompts,instructinoal=False)
                
                for sequence,prompt in zip(post_edit_response.sequences,prompts):
                    decoded_post_edit_response.append(decode_output_and_log(tokenizer=tokenizer,output=sequence,question=prompt,pre_edit=False))
                    
            post_edit_response_end_time = time.time()
            log(f"Post_edit_response inference took {post_edit_response_end_time - post_edit_response_start_time:.2f} seconds.",False,False,True)

            
            
            # Custom metric calculation
            if calculate_custom_metric_for_edited_model:
                custom_metric_array = measure_quality_sentiment_analysis(tokenizer,edited_model,edit_args)
                save_as_json(custom_metric_array,"post_custom_metric")
            


        # Unload edited model if not used later
        if not enable_models_check and not freely_chat_with_post_edit_model and editing_method != "IKE":
            del edited_model
            torch.cuda.empty_cache()
            log("Unloaded edited model",False,False,True)

    
    
    
    # Evaluate pre edit model and debugging
    if show_pre_edit_answer or enable_models_check or calculate_custom_metric_for_base_model:
        
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
            
            pre_edit_response_start_time = time.time()
            pre_edit_response = create_response(pre_edit_model,tokenizer,prompts,instructinoal=False)
            
            for sequence, prompt in zip(pre_edit_response.sequences,prompts):
                decoded_pre_edit_response.append(decode_output_and_log(tokenizer=tokenizer,output=sequence,question=prompt,pre_edit=True))
            
            pre_edit_response_end_time = time.time()
            log(f"Pre_edit_response inference took {pre_edit_response_end_time - pre_edit_response_start_time:.2f} seconds.",False,False,True)
            
            scores_string = ""
            if enable_output_scores:
                scores_string = output_scores_of_generation(tokenizer,pre_edit_response.scores,top_k)
            
            write_output_to_file("pre_edit",False,*decoded_pre_edit_response,scores_string)
        
        
        
        # Custom metric calculation
        if calculate_custom_metric_for_edited_model:
                custom_metric_array = measure_quality_sentiment_analysis(tokenizer,pre_edit_model,edit_args)
                save_as_json(custom_metric_array,"pre_custom_metric")
        
        
        
        
        # Unload if not used later
        if not enable_models_check or editing_method == "IKE" or editing_method == "INSTRUCTION_ENGINEERING":
            del pre_edit_model
            torch.cuda.empty_cache()
            log("Unloaded base model",False,False,True)
    
    
    
    
    
    # Add log info
    log_info,scores_string,models_check_string = [] , "", ""
    
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
    
    global calculate_custom_metric_for_base_model, calculate_custom_metric_for_edited_model, number_of_norms_to_edit, enable_models_check, enable_analytics, enable_output_scores, top_k, train, apply_edit, decoding_strategy, device, no_repeat_ngram_size, early_stopping, do_sample, num_beams, max_length, weights_dtype, editing_method, model_name, show_pre_edit_answer,show_post_edit_answer, freely_chat_with_post_edit_model, max_new_tokens, seed, hparams_path, train_hparams_path
    
    parser = argparse.ArgumentParser(description="Model Editing Script")
    
    parser.add_argument("-e","--editing_method", type=str, default="No editing", choices=list(available_editing_methods.values()),
                        help="Editing method to use")
    parser.add_argument("-m","--model_name", type=str, default=available_models[10],
                        help="Name of the model to use")
    parser.add_argument("-n","--number_of_norms_to_edit", type=int, default=3,
                        help="Number of norms to edit")
    parser.add_argument("-r","--show_pre_edit_answer", action="store_true",
                        help="Whether to show pre-edit answer")
    parser.add_argument("-o","--show_post_edit_answer", action="store_true",
                        help="Whether to show post-edit answer")
    parser.add_argument("-f","--freely_chat", action="store_true",
                        help="Whether to freely chat with the post-edit model")
    parser.add_argument("-t","--train", action="store_true",
                        help="Train the algorithm")
    parser.add_argument("-s","--seed", type=int, default=-1,
                        help="Random seed for reproducibility")
        
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
    
    parser.add_argument("-a","--enable_analytics", action="store_true",
                        help="Show the KL divergence and more")
    parser.add_argument("-c","--custom_metric", action="store_true",
                        help="Acitvate the custom metric calculation for edited model")
    parser.add_argument("-b","--custom_metric_base", action="store_true",
                        help="Acitvate the custom metric calculation for base model")
    parser.add_argument("--enable_cpu_inference", action="store_true",
                        help="Whether to do the inference on the CPU")
    parser.add_argument("--enable_models_check", action="store_true",
                        help="Check whether the post_edit model did change")
    parser.add_argument("--enable_output_scores", action="store_true",
                        help="Show the scores for the most probable tokens")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top k probable tokens for the output scores")
    
    parser.add_argument("-w",'--weights_dtype', type=str, choices=['float32', 'float16', 'bfloat16'],
                        default='float16', help='Data type for weights: float32, 16 or bfloat16' )
    parser.add_argument("--config_file_name", type=str, default=available_models[10].split("/")[1],
                        help="Name of the config file")
    
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
    calculate_custom_metric_for_edited_model = args.custom_metric
    calculate_custom_metric_for_base_model = args.custom_metric_base
        
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
    print(f"{'Number of norms to edit:':<{col_width}} {number_of_norms_to_edit}")
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