from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from transformers import set_seed
import torch
import time
import argparse
import warnings 
import os
from colorama import Fore, Back, Style, init

import config
from utils import *
from evaluation import *
from edit import edit


# TODO:


# Final touches in cutsom metric
# Calculate KL div for locality


# Try backtranslation technique using vllm   Done
# Try causal tracing and argument that it is not useful since every we only have a rot which is every time of different structure Done
# We have no subjects  Done

# Fix IKE for norms  
# Add situation to prompt in the edit norms dataset  Done
# Add locality prompts as an original norm   Done

# Fix the edit norms dataset   Done
## Change the structure       Done


# Generate rephrases using o3 api or using tubs   #DONE
# Find out what to do with subject for ROME  --> Pick for example the action always???    Done


# Find out the difference between locality neighborhood and locality distracting     #DONE
## locality neighborhood are prompts with the same relation and object as the edit but different subjects 

# These neighborhood
# prompts can be used to inspect whether the model   
# edit has undesired side effects on closely related
# factual associations

## locality distrcting neighborhood is similar with a distracting statement at the beginning which is the edited prompt
## Relations in facts are based on the wikidata relation types and there are too many
## In our case of norms we only have one relation or one type of statements

# Add api icl chatgpt4     # Not really necessary

# Try instead of rephrasing backtranslation technique  Done

# Change loading in edit        Done
# Fix generating edit norms dataset and fix locality   Done



# Maybe fix subject difference in im/moral actions and ground_truth

# For the custom metric cut the response till the point



# Trian MEND on the norms dataset

# write an extra file for the difference of the evaluations
# To take into account the initial knowledge of the model

# Make the locality prompts those that have contrary action judgement
# not necessarily good btw
# Add KL divergence to the final locality

# For example generate many sequences and calculate average or search for a specific word and see what it's probability is
# or take a look at the first 20 words and take the average of their sentiment


# Take a look on how they implemented counterfact, it it based on model's knowledge??  NO


# First of all fix IKE and then add causal tracing  









# TODO:

# Necessary:
# Fix num_return_sequences and extend it for every metric  done
# Make the output format more consistent    Done
# Fix temperature and try it     
# Extend the KL divergence implementation done
# Add the different decoding strategy names done
# Finish the visualization plots
# Count or measure NS and NM
# Implement perplexity    Done
# Take config parameters as default ones  Done

# First do all pre edit model things
# then edit
# then evaluate post edits
# KL Div has an issue
# Do kl div for first token + differing token
# In the plot later, output the average of those above also

# Optional:
# Implement the causal tracing.
# Finish the subject finding implementation
# Add the new templates to edit dataset.
# Implement the metric of Niklas.
# Train MEND on a split/half or 90% of our edit dataset.
# Finish the instruction engineering method.
# Try the different decoding strategies.
# Change moral coherence
# Add beams number to file and plot metrics with that
# Change perplexity to be not abs but the percentage of change
# Ask if we should not take value of 0-token if index is -1
# Add arrows to the metrics
# Split into multiple tables

# Plots:
# Plot locality + perplexity with sequential edits number








def main():
    
    
    # ---------------------------------------------------------------- #
    # ---------------- Some Initialization Stuff --------------------- #
    # ---------------------------------------------------------------- #
    
    init()
    os.makedirs('outputs/', exist_ok=True)

    # Ignore all warnings
    warnings.filterwarnings("ignore")
    
    if not torch.cuda.is_available() and config.device != torch.device('cpu'):
        print("Torch cuds is not available")
        return
    
    login(token=config.access_token,add_to_git_credential=True)
    
    if config.seed != -1:
        set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    tokenizer.padding_side='left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        


    # Load the edit norms dataset
    norms_dict = load_norms()
    pre_edit_model, post_edit_model = None, None



    # ---------------------------------------------------------------- #
    # --------------------- Evaluating Process ----------------------- #
    # ------------------ Evaluating pre_edit_model ------------------- #
    # ---------------------------------------------------------------- #    
    
    
    # Custom metrics calculation for pre_edit_model
 
    # Load the pre_edit_model
    pre_edit_model = load_pre_edit_model()
        
    # All needed outputs for pre_edit_model
    pre_edit_output_dict, pre_edit_logits_dict, pre_edit_scores_dict, _ = preprare_responses(tokenizer, pre_edit_model, True, norms_dict, None)
    
    # Write pre_edit_response to a file
    save_as_json(norms_dict | pre_edit_output_dict, "pre_edit_logs")

    if config.enable_sentiment:
        pre_edit_sentiment = calculate_sentiment_analysis_labels(norms_dict, True, pre_edit_output_dict, None)
        save_as_json(pre_edit_sentiment, "pre_edit_sentiment")
    
    if config.enable_perplexity:
        pre_edit_perplexity = calculate_perplexity_for_locality(tokenizer, pre_edit_model, pre_edit_output_dict)
        save_as_json(pre_edit_perplexity, "pre_edit_perplexity")

    # Unload pre_edit_model if not used later
    unload_pre_edit_model(pre_edit_model)




    # ---------------------------------------------------------------- #
    # --------------------- Editing Process -------------------------- #
    # ---------------------------------------------------------------- #
    
    
        
    # Initialize the arguments dictionary
    edit_args = {
        "prompts": norms_dict["prompts"],
        "ground_truth": norms_dict["ground_truth"],
        "target_new": norms_dict["target_new"],
        "subject": norms_dict["subject"],
        "locality_inputs": norms_dict["locality_inputs"],
        "locality_inputs_action_moral_judgement" : norms_dict["locality_inputs_action_moral_judgement"],
        "rephrase_prompts": norms_dict["strong_rephrase_prompts"],
        "portability_inputs": norms_dict["portability_inputs"],
        "loc_prompts" : norms_dict["loc_prompts"],
        "moral_action": norms_dict["moral_action"],
        "immoral_action": norms_dict["immoral_action"],
        "action_moral_judgment": norms_dict["action_moral_judgment"],
        "light_rephrase_prompts": norms_dict["light_rephrase_prompts"],
        "strong_rephrase_prompts": norms_dict["strong_rephrase_prompts"],
        "sequential_edit": True
    }
            
    
    post_edit_easy_edit_metrics, post_edit_model, ike_generation_prompts, editing_time = edit(edit_args, tokenizer)    
              
    
    if config.train:
        log(f"Training took {editing_time:.2f} seconds.",False,False,True)
        return
    else:
        log(f"Editing took {editing_time:.2f} seconds.",False,False,True)
        
            
         
         
    # Load the pre_edit_model
    if config.editing_method == "IKE":
        post_edit_model = pre_edit_model
    else:
        # Saving the post edit metrics of Easy Edit 
        save_as_json(post_edit_easy_edit_metrics,"post_edit_easy_edit_metrics")
        log("Metrics saved as json file",False,False,False)
        log("Loaded edited model",True,False,True)
        print_gpu_memory()     
         
    
          
         
    # ---------------------------------------------------------------- #
    # --------------------- Evaluating process ----------------------- #
    # ----------------- Evaluating post_edit_model ------------------- #
    # ---------------------------------------------------------------- #
        
     
    
    # All needed outputs for post_edit_model
    post_edit_output_dict, post_edit_logits_dict, post_edit_scores_dict, new_ike_edit_args = preprare_responses(tokenizer, post_edit_model, False , norms_dict, ike_generation_prompts)
    
    # Write post_edit_response to a file
    if config.editing_method == "IKE":
        save_as_json(new_ike_edit_args | post_edit_output_dict,"post_edit_logs")
    else:
        save_as_json(norms_dict | post_edit_output_dict,"post_edit_logs")

    
    # Calculate the custom metrics for post_edit_model
    if config.enable_sentiment:
        post_edit_sentiment = calculate_sentiment_analysis_labels(norms_dict, False, post_edit_output_dict, new_ike_edit_args)
        save_as_json(post_edit_sentiment,"post_edit_sentiment")
    
    
    if config.enable_perplexity:
        post_edit_perplexity = calculate_perplexity_for_locality(tokenizer, post_edit_model, pre_edit_output_dict)
        save_as_json(post_edit_perplexity, "post_edit_perplexity")
    
    
        
    # Unload post_edit_model if not used later
    unload_post_edit_model(post_edit_model)
        
        
        
        
    
    
    # ---------------------------------------------------------------- #
    # --------------------- Evaluating process ----------------------- #
    # ----------------- Measuring the Edit Effects ------------------- #
    # ---------------------------------------------------------------- #
    
    
    
    # Show the effects of the edit
    if config.enable_sentiment:
        edit_effect_sentiment_metric = evaluate_sentiment_metric(pre_edit_sentiment, post_edit_sentiment)
        save_as_json(edit_effect_sentiment_metric,"edit_effect_sentiment_metric")
    
    if config.enable_perplexity:
        edit_effect_perplexity_metric = evaluate_perplexity_metric(pre_edit_perplexity, post_edit_perplexity)
        save_as_json(edit_effect_perplexity_metric,"edit_effect_perplexity_metric")
    
    if config.enable_kl_div:
        edit_effect_kl_div_metric = evaluate_kl_div_metric(tokenizer, pre_edit_logits_dict, post_edit_logits_dict, pre_edit_output_dict, post_edit_output_dict, norms_dict)
        save_as_json(edit_effect_kl_div_metric,"edit_effect_kl_div_metric")
    

 
      
    
      
    # ---------------------------------------------------------------- #
    # --------------------- Debugging process ------------------------ #
    # ---------------------------------------------------------------- #
    
    # Output scores, KL divergence and other useful information
    output_debugging_info(tokenizer, pre_edit_model, post_edit_model, edit_args, pre_edit_output_dict, post_edit_output_dict, pre_edit_logits_dict, post_edit_logits_dict, pre_edit_scores_dict, post_edit_scores_dict)


    # Freely chat with the post edit model
    if config.freely_chat_with_post_edit_model:
        chat_with_model(post_edit_model,tokenizer)





def parse_arguments():
    
    parser = argparse.ArgumentParser(description="Model Editing Script")
    
    # Shortcuts : e,s,f,t,m,n,d,k,o,w,a,b,c
    
    parser.add_argument("-e","--editing_method", type=str, default="No editing", choices=list(config.available_editing_methods.values()),
                        help="Editing method to use\nIf not specified, then no editing is performed")
    parser.add_argument("--model_name", type=str, default=config.model_name,
                        help="Name of the model to use")
    parser.add_argument("-s","--norms_subset_size", type=int, default=config.norms_subset_size,
                        help="Number of norms to edit")
    parser.add_argument("-f","--freely_chat", action="store_true",
                        help="Whether to freely chat with the post-edit model")
    parser.add_argument("-t","--train", action="store_true",
                        help="Train the algorithm")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle the dataset")
    
    # Decoding strategy parameters
    parser.add_argument("--seed", type=int, default=config.seed,
                        help="Random seed for reproducibility, leave -1 for same hardcoded seed always")
    parser.add_argument("--max_length", type=int, default=config.max_length,
                        help="Maximum number of tokens in the prompt")
    parser.add_argument("-m", "--max_new_tokens", type=int, default=config.max_new_tokens,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("-n", "--num_beams", type=int, default=config.num_beams,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("-d", "--do_sample", action="store_true",
                        help="Activate multinomial-sampling")
    parser.add_argument("--early_stopping", action="store_true",
                        help="Early stopping")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=config.no_repeat_ngram_size,
                        help="No repeat ngram size")
    
    # Debugging stuff
    parser.add_argument("-k", "--enable_analytics", action="store_true",
                        help="Show the KL divergence and more")
    parser.add_argument("--enable_models_check", action="store_true",
                        help="Check whether the post_edit model did change")
    
    # Extra information
    parser.add_argument("-o", "--enable_output_scores", action="store_true",
                        help="Show the scores for the most probable tokens")
    parser.add_argument("--scores_top_k", type=int, default=config.scores_top_k,
                        help="Top k probable tokens for the output scores")
    

    # Evaluation
    parser.add_argument("-a", "--enable_sentiment", action="store_true",
                        help="Enable sentiment calculation")
    parser.add_argument("-b", "--enable_perplexity", action="store_true",
                        help="Enable perplexity calculation")
    parser.add_argument("-c", "--enable_kl_div", action="store_true",
                        help="Enable kl divergence calculation")
  
    
    
    parser.add_argument("--enable_cpu_inference", action="store_true",
                        help="Whether to do the inference on the CPU")
    parser.add_argument("-w",'--weights_dtype', type=str, choices=['float32', 'float16', 'bfloat16'],
                        default='float32', help='Data type for weights: float32, 16 or bfloat16' )
    parser.add_argument("--config_file_name", type=str, default=config.model_name.split("/")[1],
                        help="Name of the config file")
    
    args = parser.parse_args()
    
    
    
    # Update global variables
    config.editing_method = args.editing_method
    config.model_name = args.model_name
    config.freely_chat_with_post_edit_model = args.freely_chat
    config.norms_subset_size = args.norms_subset_size
    
    args.config_file_name = config.model_name.split("/")[1]
    
    config.hparams_path =  os.path.join(get_ml_path(), "hparams", config.editing_method, f"{args.config_file_name}.yaml")
    config.train_hparams_path = os.path.join(get_ml_path(), "hparams", "TRAINING", config.editing_method, f"{args.config_file_name}.yaml")
    
    dtype_map = { 'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16 }
    config.weights_dtype = dtype_map[args.weights_dtype]
    
    config.seed = args.seed
    config.max_new_tokens = args.max_new_tokens
    config.max_length = args.max_length
    config.num_beams = args.num_beams
    config.no_repeat_ngram_size = args.no_repeat_ngram_size
    config.early_stopping = args.early_stopping
    config.do_sample = args.do_sample
    config.scores_top_k = args.scores_top_k
    config.apply_edit = True
    config.train = args.train
    config.enable_analytics = args.enable_analytics
    config.enable_output_scores = args.enable_output_scores
    config.enable_models_check = args.enable_models_check
    config.num_return_sequences = config.num_beams
    config.enable_sentiment = args.enable_sentiment
    config.enable_kl_div = args.enable_kl_div
    config.enable_perplexity = args.enable_perplexity
    
    
    config.decoding_strategy = "greedy-decoding"
    
    if config.num_beams == 1 and config.do_sample == False:
        config.decoding_strategy = "greedy-decoding"
        
    elif config.num_beams > 1 and config.do_sample == False:
        config.decoding_strategy = "beam-search"
        
    elif config.num_beams > 1 and config.do_sample == True:
        config.decoding_strategy = "beam-search multinomial sampling"    
        
    else:
        config.decoding_strategy = "multinomial-sampling"
        
        
        
    
    print(config.decoding_strategy)
    
    if not args.enable_cpu_inference:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')
    
    
    config.shuffle = args.shuffle
    
    
    col_width = 27
    
    print()
    print('-'*75)
    
    print(Fore.BLUE)
    print("Edit Configuration")
    print(f"{'Model_name:':<{col_width}} {config.model_name}")
    print(f"{'Editing_method:':<{col_width}} {config.editing_method}")
    print(f"{'Decoding_strategy:':<{col_width}} {config.decoding_strategy}")
    print(f"{'Number of norms to edit:':<{col_width}} {config.norms_subset_size}")
    print(f"{'Device:':<{col_width}} {str(config.device)}")
    
    print(Fore.LIGHTYELLOW_EX)
    print("Information to Output")
    print(f"{'train:':<{col_width}} {str(config.train)}")
    
    print(Fore.CYAN)
    print("Debugging Informations")
    print(f"{'enable_output_scores:':<{col_width}} {str(config.enable_output_scores)}")
    print(f"{'enable_analytics:':<{col_width}} {str(config.enable_analytics)}")
    print(f"{'enable_models_check:':<{col_width}} {str(config.enable_models_check)}") 
    print(f"{'freely chat with model:':<{col_width}} {str(config.freely_chat_with_post_edit_model)}")
    
    print(Fore.LIGHTRED_EX)
    print("Extra Configuration")
    print(f"{'weights_dtype:':<{col_width}} {str(config.weights_dtype)}")
    print(f"{'hparams_path:':<{col_width}} {config.hparams_path}")
    print(f"{'available_gpu_memory:':<{col_width}} {str(get_available_gpu_memory())}")
    print(Style.RESET_ALL)
    print('-'*75)
    print()
    
    return args


if __name__ == '__main__':
    
    parse_arguments()
    main()