import torch
from transformers import pipeline
from datasets import load_dataset
from colorama import Fore, Back, Style, init
import statistics
import config
from utils import create_response
from utils import log, write_output_to_file
from dataset_creation.rephrases.utils import send_request






def load_norms():
    
    dataset_path = f"{config.datasets_path}/norms/edit_norms_datasets/edit_norms_dataset.json"
    
    full_dataset = load_dataset("json", data_files = dataset_path, split='train')
    
    if config.shuffle:
        full_dataset = full_dataset.shuffle()

    config.norms_subset_size =  min(config.norms_subset_size, len(full_dataset) // 2)
    
    ds = full_dataset.select(range(config.norms_subset_size))
    locality_dataset = full_dataset.select(range(config.norms_subset_size, config.norms_subset_size + config.norms_subset_size))
    
    prompts = ds['prompt']
    ground_truth = ds['ground_truth']
    target_new = ds['target_new']
    action_moral_judgment = ds["action_moral_judgment"]
    
    subject = ds['subject']
    light_rephrase_prompts = ds['light_rephrase_prompts']
    strong_rephrase_prompts = ds['strong_rephrase_prompt']
    prompt_subject = ds['prompt_subject']
    
    moral_action = ds["moral_action"]
    immoral_action = ds["immoral_action"]
    situation = ds["situation"]
    
    
    locality_inputs_neighborhood_prompt = locality_dataset["prompt"]
    locality_inputs_neighborhood_ground_truth = locality_dataset["ground_truth"]
    locality_inputs_action_moral_judgement = locality_dataset["action_moral_judgment"]
    
    locality_inputs_distracting_prompt = [f"{prompts[i]} {target_new[i]}. {locality_inputs_neighborhood_prompt[i]}" for i in range(len(locality_dataset))]
    locality_inputs_distracting_ground_truth = locality_dataset["ground_truth"]

    portability_inputs_one_hop_prompt = [f"{sit} {mor}" for sit, mor in zip(situation, moral_action)]
    portability_inputs_two_hop_prompt = [f"{sit} {immor}" for sit, immor in zip(situation, immoral_action)]
    
    # Needed for WISE
    loc_prompts = [f"{pr} {tn}" for pr, tn in zip(locality_inputs_neighborhood_prompt, locality_inputs_neighborhood_ground_truth)]

    locality_inputs = {}
    portability_inputs = {}
    
    locality_inputs = {
        "neighborhood":{
            "prompt": locality_inputs_neighborhood_prompt,
            "ground_truth": locality_inputs_neighborhood_ground_truth
        },
        "distracting":{
            "prompt": locality_inputs_distracting_prompt,
            "ground_truth": locality_inputs_distracting_ground_truth
        }
	}         

             
    portability_inputs = {
        "synonym":{
            "prompt": prompt_subject,
            "ground_truth":target_new
        },
      
        "one_hop":{
            "prompt": portability_inputs_one_hop_prompt,
            "ground_truth": target_new
        },
        
        "two_hop":{
            "prompt": portability_inputs_two_hop_prompt,
            "ground_truth": ground_truth
        }
    }         

        
    # Check whether locality and portability are empty
    log("Norms dataset loaded",False,False,True)
    
    norms_dict = {
        "prompts": prompts,
        "ground_truth": ground_truth,
        "target_new": target_new,
        "subject": subject,
        "light_rephrase_prompts": light_rephrase_prompts,
        "strong_rephrase_prompts": strong_rephrase_prompts,
        "locality_inputs": locality_inputs,
        "locality_inputs_action_moral_judgement": locality_inputs_action_moral_judgement,
        "portability_inputs": portability_inputs,
        "action_moral_judgment": action_moral_judgment,
        "moral_action": moral_action,
        "immoral_action": immoral_action, 
        "loc_prompts": loc_prompts
    }

    return norms_dict








def output_token_scores(tokenizer, scores, batch_element_index):
    
    # Get top 10 tokens and their probabilities
    score_output = ""
    top_tokens = []
    for score in scores:
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(score, dim=-1)
        # Get the top 10 tokens
        top_k_probs, top_k_ids = torch.topk(probs, config.top_k, dim=-1)
        top_tokens.append((top_k_ids, top_k_probs))


    col1_width = 30
    col2_width = 30
    
    # Decode tokens to strings and print
    for i, (ids, probs) in enumerate(top_tokens):
        score_output += f"Token {i + 1}:\n"
        print(Fore.LIGHTMAGENTA_EX + f"Token {i + 1}:" + Style.RESET_ALL)
        
        for token, prob in zip(ids[batch_element_index], probs[batch_element_index]):
            
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




def analyse_kl_divergence(pre_edit_logits,post_edit_logits) -> str:
    output = ""
    if pre_edit_logits and post_edit_logits and config.editing_method != "IKE":
        kl_div_first_token = calculate_kl_divergence_for_token(pre_edit_logits,post_edit_logits, 2)
        kl_div_all_tokens, biggest_div, biggest_div_index = calculate_kl_divergence_amongst_all_tokens(pre_edit_logits,post_edit_logits)
        check2 = f"KL divergence for first token: {kl_div_first_token}"
        check3 = f"KL divergence amongst all tokens: {kl_div_all_tokens}"
        check4 = f"Biggest KL divergence is on token {biggest_div_index} with the value of {biggest_div}"
        log(check2,True,True,True)
        log(check3,True,True,True)
        log(check4,True,True,True)
        output += check2 + "\n" + check3 + "\n" + check4 + "\n"

    return output








def calculate_kl_divergence_for_token(pre_edit_logits, post_edit_logits, token_index):
    
    # Pick the first token
    pre_edit_logit = pre_edit_logits[token_index]
    post_edit_logit = post_edit_logits[token_index]
    
    # Move to same device
    pre_edit_logit = pre_edit_logit.to(post_edit_logit.device)
    
    # Convert logits to probabilities
    original_probs = torch.nn.functional.softmax(pre_edit_logit, dim=-1)
    edited_probs = torch.nn.functional.softmax(post_edit_logit, dim=-1)

    # Compute KL divergence
    kl_divergence = torch.nn.functional.kl_div(original_probs.log(), edited_probs, reduction='batchmean')

    return kl_divergence






def calculate_kl_divergence_amongst_all_tokens(pre_edit_logits, post_edit_logits):
    result = 0
    biggest_kl_divergence = 0
    biggest_kl_divergence_index = 0
    
    for i in range(len(pre_edit_logits)):
        current_kl_divergence = calculate_kl_divergence_for_token(pre_edit_logits,post_edit_logits, i).item()
        result += current_kl_divergence

        if current_kl_divergence > biggest_kl_divergence:
            biggest_kl_divergence = current_kl_divergence
            biggest_kl_divergence_index = i
        
    return result, biggest_kl_divergence, biggest_kl_divergence_index






def preprare_responses(tokenizer, pre_edit_model, post_edit_model, edit_args):
    
    model = pre_edit_model
    
    if model is None:
        model = post_edit_model
        

    # Gets the generated part, strips unnecessary characters from the left and return the string till the period
    def format_output(output, length):
        result = output[length:]
        result = result.lstrip(" .,?\n")
        p_index = result.find('.') 
        q_index = result.find('?')
        
        if p_index != -1:
            result = f"{result[:p_index]}."
        elif q_index!= -1:
            result = f"{result[:q_index]}?"
        else:
            result = result
              
        return result    
    
     
    
    decoded_responses_prompt = []
        
    decoded_responses_light_rephrase_1 = []
    decoded_responses_light_rephrase_2 = []
    decoded_responses_light_rephrase_3 = []
    
    decoded_responses_strong_rephrase = []
    
    decoded_responses_portability_synonym = []
    decoded_responses_portability_one_hop = [] 
    decoded_responses_portability_two_hop = [] 

    decoded_responses_locality_neighborhood = []
    decoded_responses_locality_distracting = []

    logits = []
    scores = []
    
    for index in range(0, len(edit_args["prompts"])):
        
        # To work all answers in parallel
        model_input = [
            
            edit_args["prompts"][index],
                   
            edit_args["light_rephrase_prompts"][index][0],
            edit_args["light_rephrase_prompts"][index][1],
            edit_args["light_rephrase_prompts"][index][2],
            
            edit_args["strong_rephrase_prompts"][index],
            
            edit_args["portability_inputs"]["synonym"]["prompt"][index],
            edit_args["portability_inputs"]["one_hop"]["prompt"][index],
            edit_args["portability_inputs"]["two_hop"]["prompt"][index],
    
            edit_args["locality_inputs"]["neighborhood"]["prompt"][index],
            edit_args["locality_inputs"]["distracting"]["prompt"][index],

        ]
        
        
        # Create responses then batch decode then reformat the outputs
        output = create_response(model,tokenizer,model_input,instructinoal=False)
        decoded_output = tokenizer.batch_decode(output.sequences,skip_special_tokens=True)
        
        decoded_responses_prompt.append(format_output(decoded_output[0], len(edit_args["prompts"][index])))
        
        decoded_responses_light_rephrase_1.append(format_output(decoded_output[1], len(edit_args["light_rephrase_prompts"][index][0])))
        decoded_responses_light_rephrase_2.append(format_output(decoded_output[2], len(edit_args["light_rephrase_prompts"][index][1])))
        decoded_responses_light_rephrase_3.append(format_output(decoded_output[3], len(edit_args["light_rephrase_prompts"][index][2])))
        
        decoded_responses_strong_rephrase.append(format_output(decoded_output[4], len(edit_args["strong_rephrase_prompts"][index]))) 
        
        decoded_responses_portability_synonym.append(format_output(decoded_output[5], len(edit_args["portability_inputs"]["synonym"]["prompt"][index]))) 
        decoded_responses_portability_one_hop.append(format_output(decoded_output[6], len(edit_args["portability_inputs"]["one_hop"]["prompt"][index]))) 
        decoded_responses_portability_two_hop.append(format_output(decoded_output[7], len(edit_args["portability_inputs"]["two_hop"]["prompt"][index])))

        decoded_responses_locality_neighborhood.append(format_output(decoded_output[8], len(edit_args["locality_inputs"]["neighborhood"]["prompt"][index]))) 
        decoded_responses_locality_distracting.append(format_output(decoded_output[9], len(edit_args["locality_inputs"]["distracting"]["prompt"][index]))) 

        logits.append(output.logits)
        scores.append(output.scores)

    return_dict = {
        "prompt": decoded_responses_prompt,
        "light_rephrase_1": decoded_responses_light_rephrase_1,
        "light_rephrase_2": decoded_responses_light_rephrase_2,
        "light_rephrase_3": decoded_responses_light_rephrase_3,
        "strong_rephrase": decoded_responses_strong_rephrase,
        "portability_synonym": decoded_responses_portability_synonym,
        "portability_one_hop": decoded_responses_portability_one_hop,
        "portability_two_hop": decoded_responses_portability_two_hop,
        "locality_neighborhood": decoded_responses_locality_neighborhood,
        "locality_distracting": decoded_responses_locality_distracting,
    }
    

    logits_dict = {
        "prompt": tuple(torch.cat( tuple(tup[i][0] for tup in logits), dim=0).reshape(len(logits), tokenizer.vocab_size) for i in range(len(logits[0]))),
        "light_rephrase_1": tuple(torch.cat( tuple(tup[i][1] for tup in logits), dim=0).reshape(len(logits), tokenizer.vocab_size) for i in range(len(logits[0]))),
        "light_rephrase_2": tuple(torch.cat( tuple(tup[i][2] for tup in logits), dim=0).reshape(len(logits), tokenizer.vocab_size) for i in range(len(logits[0]))),
        "light_rephrase_3": tuple(torch.cat( tuple(tup[i][3] for tup in logits), dim=0).reshape(len(logits), tokenizer.vocab_size) for i in range(len(logits[0]))),
        "strong_rephrase": tuple(torch.cat( tuple(tup[i][4] for tup in logits), dim=0).reshape(len(logits), tokenizer.vocab_size) for i in range(len(logits[0]))),
        "portability_synonym": tuple(torch.cat( tuple(tup[i][5] for tup in logits), dim=0).reshape(len(logits), tokenizer.vocab_size) for i in range(len(logits[0]))),
        "portability_one_hop": tuple(torch.cat( tuple(tup[i][6] for tup in logits), dim=0).reshape(len(logits), tokenizer.vocab_size) for i in range(len(logits[0]))),
        "portability_two_hop": tuple(torch.cat( tuple(tup[i][7] for tup in logits), dim=0).reshape(len(logits), tokenizer.vocab_size) for i in range(len(logits[0]))),
        "locality_neighborhood": tuple(torch.cat( tuple(tup[i][8] for tup in logits), dim=0).reshape(len(logits), tokenizer.vocab_size) for i in range(len(logits[0]))),
        "locality_distracting": tuple(torch.cat( tuple(tup[i][9] for tup in logits), dim=0).reshape(len(logits), tokenizer.vocab_size) for i in range(len(logits[0]))),
    }
    
    scores_dict = {
        "prompt": tuple(torch.cat( tuple(tup[i][0] for tup in scores), dim=0).reshape(len(scores), tokenizer.vocab_size) for i in range(len(scores[0]))),
        "light_rephrase_1": tuple(torch.cat( tuple(tup[i][1] for tup in scores), dim=0).reshape(len(scores), tokenizer.vocab_size) for i in range(len(scores[0]))),
        "light_rephrase_2": tuple(torch.cat( tuple(tup[i][2] for tup in scores), dim=0).reshape(len(scores), tokenizer.vocab_size) for i in range(len(scores[0]))),
        "light_rephrase_3": tuple(torch.cat( tuple(tup[i][3] for tup in scores), dim=0).reshape(len(scores), tokenizer.vocab_size) for i in range(len(scores[0]))),
        "strong_rephrase": tuple(torch.cat( tuple(tup[i][4] for tup in scores), dim=0).reshape(len(scores), tokenizer.vocab_size) for i in range(len(scores[0]))),
        "portability_synonym": tuple(torch.cat( tuple(tup[i][5] for tup in scores), dim=0).reshape(len(scores), tokenizer.vocab_size) for i in range(len(scores[0]))),
        "portability_one_hop": tuple(torch.cat( tuple(tup[i][6] for tup in scores), dim=0).reshape(len(scores), tokenizer.vocab_size) for i in range(len(scores[0]))),
        "portability_two_hop": tuple(torch.cat( tuple(tup[i][7] for tup in scores), dim=0).reshape(len(scores), tokenizer.vocab_size) for i in range(len(scores[0]))),
        "locality_neighborhood": tuple(torch.cat( tuple(tup[i][8] for tup in scores), dim=0).reshape(len(scores), tokenizer.vocab_size) for i in range(len(scores[0]))),
        "locality_distracting": tuple(torch.cat( tuple(tup[i][9] for tup in scores), dim=0).reshape(len(scores), tokenizer.vocab_size) for i in range(len(scores[0])))
    }
    
    
    return return_dict, logits_dict, scores_dict
   
   
   







# A custom metric that measures the quality using sentiment_analysis and KL divergence
def measure_quality_sentiment_analysis(edit_args, pre_edit, output_dict):
        
    # Test other metrics after editing to see whether model is degraded    
    negative_label = "LABEL_0"
    positive_label = "LABEL_2"
    neutral_label = "LABEL_1"
    
        
    # Compare with expected label and return float as score
    # I decided to take a neutral label as 0.5
    def labels_to_float(actual_label, target_label):
        if actual_label == neutral_label:
            return 0.5
        else:
            return float(target_label == actual_label)
        
        
    def compare_labels(actual_labels, target_label):
        if isinstance(actual_labels, str):
            return labels_to_float(actual_labels, target_label)
        elif isinstance(actual_labels,list):
            new_list = [labels_to_float(i,target_label) for i in actual_labels]
            return statistics.mean(new_list)
        else:
            return None
    
        
    def float_to_label(value):
        if value < 0:
            return negative_label
        elif value == 0:
            return neutral_label
        else:
            return positive_label
    

    def invert_label(label):
        if label == positive_label:
            return negative_label
        elif label == negative_label:
            return positive_label
        else:
            return neutral_label
        


    # "siebert/sentiment-roberta-large-english"
    # "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    sentiment_analysis = pipeline("sentiment-analysis", model=model_name, device=0)
    
    custom_metric_array = []
    

    for index in range(0,len(edit_args["prompts"])):
        

        # To work all answers in parallel
        sentiment_input = [
            edit_args["ground_truth"][index],
            edit_args["target_new"][index],
            output_dict["prompt"][index],
            output_dict["light_rephrase_1"][index],
            output_dict["light_rephrase_2"][index],
            output_dict["light_rephrase_3"][index],
            output_dict["strong_rephrase"][index],
            output_dict["portability_synonym"][index],
            output_dict["portability_one_hop"][index],
            output_dict["portability_two_hop"][index],
            output_dict["locality_neighborhood"][index],
            output_dict["locality_distracting"][index]
        ]
        
        
        # Generate predictions
        sentiment_output = sentiment_analysis(sentiment_input, batch_size=len(sentiment_input))
        
        target_new_label = float_to_label(int(edit_args["action_moral_judgment"][index]))
        ground_truth_label = invert_label(target_new_label)

        generated_ground_truth_label = sentiment_output[0]["label"]
        generated_target_new_label = sentiment_output[1]["label"]
        
        sentiment_analysis_model_reliability = statistics.mean([compare_labels(ground_truth_label, generated_ground_truth_label), compare_labels(target_new_label, generated_target_new_label)])
        
        # If testing the pre_edit model then invert the labels
        if pre_edit:
            target_new_label = invert_label(target_new_label)
        
        reliability = compare_labels(sentiment_output[2]["label"], target_new_label)
        light_generality = compare_labels([sentiment_output[3]["label"], sentiment_output[4]["label"], sentiment_output[5]["label"]], target_new_label)
        strong_generality = compare_labels(sentiment_output[6]["label"], target_new_label)

        portability_synonym = compare_labels(sentiment_output[7]["label"], target_new_label)
        portability_one_hop = compare_labels(sentiment_output[8]["label"], target_new_label)
        portability_two_hop = compare_labels(sentiment_output[9]["label"], ground_truth_label)
        
        expected_locality_neighborhood_label = invert_label(float_to_label(edit_args["locality_inputs_action_moral_judgement"][index]))
        
        locality_neighborhood = compare_labels(sentiment_output[10]["label"], expected_locality_neighborhood_label)
        locality_distracting = compare_labels(sentiment_output[11]["label"], expected_locality_neighborhood_label)
        
        custom_metric = {
            "sentiment_analysis_model_reliability":sentiment_analysis_model_reliability,
            "reliability":reliability,
            "light_generality":light_generality,
            "strong_generality":strong_generality,
            "synonym_generality":portability_synonym,
            "one_hop_inference":portability_one_hop,
            "two_hop_inference":portability_two_hop,
            "locality_neighborhood":locality_neighborhood,
            "locality_distracting":locality_distracting
        }
    
        custom_metric_array.append(custom_metric)
    
    
    return custom_metric_array







# This makes sure that we take into account the initial knowledge of the model
def evaluate_edit_effect_sentiment_metric(pre_edit_custom_metric, post_edit_custom_metric):
    edit_changes_custom_metric = []
    
    def measure_edit_succes_rate(pre_edit_value, post_edit_value):
        return min(1 - (pre_edit_value - post_edit_value),1)
    
    for pre_edit_item, post_edit_item in zip(pre_edit_custom_metric, post_edit_custom_metric):
        item = {}
        for key in pre_edit_item:
            item.update({key : f"{pre_edit_item[key]:.3f} --> {post_edit_item[key]:.3f} = {measure_edit_succes_rate(pre_edit_item[key], post_edit_item[key]):.3f} = {measure_edit_succes_rate(pre_edit_item[key], post_edit_item[key])*100:.2f}%"})
        
        edit_changes_custom_metric.append(item)
    
    return edit_changes_custom_metric







def evaluate_edit_effect_kl_div_metric(pre_edit_logits_dict, post_edit_logits_dict):

    kl_div_dict = {k: calculate_kl_divergence_for_token(pre_edit_logits_dict[k], post_edit_logits_dict[k], 0).item() for k in pre_edit_logits_dict.keys()} | {f"{k}_amongst_all_tokens": calculate_kl_divergence_amongst_all_tokens(pre_edit_logits_dict[k], post_edit_logits_dict[k]) for k in pre_edit_logits_dict.keys()}
    return kl_div_dict





def measure_quality_chatgpt_api(edit_args, decoded_post_edit_response):
    template = f"{decoded_post_edit_response}"
    response = send_request(template)









# ---------------------------------------------------------------- #
# ---------------------------------------------------------------- #
# -----------------Utils for debugging purposes------------------- #
# ---------------------------------------------------------------- #
# ---------------------------------------------------------------- #




def analyse_reliability_of_edit(decoded_post_edit_response,target_new) -> str:

    output = ""
    edit_successfull =  target_new.lower() in decoded_post_edit_response.lower()
    check1 = f"Does the post_edit_answer contain the target answer? {edit_successfull}"
    log(check1,True,True,True)
    output += check1 + "\n"

    return output




def check_model_weights_changed(pre_edit_model, post_edit_model):
    if pre_edit_model and post_edit_model:
        output = "The models are the same."
        for parameter_name, parameter_value in pre_edit_model.state_dict().items():
            if not torch.equal(parameter_value, post_edit_model.state_dict()[parameter_name]):
                output = "The models are different."
                
        log(output,True,True,True)
        return output
    
    
    
    
    
    
def output_debugging_info(tokenizer, pre_edit_model, post_edit_model, edit_args, pre_edit_output_dict, post_edit_output_dict, pre_edit_logits_dict, post_edit_logits_dict, pre_edit_scores_dict, post_edit_scores_dict):
    
    # Add log info
    log_info, pre_edit_scores_string, post_edit_scores_string, models_check_string = [] , "", "", ""
    
    # Some debugging information
    if config.enable_analytics:
        # for i in range(len(decoded_post_edit_response)):
        #     log_info.append(analyse_reliability_of_edit(decoded_post_edit_response=decoded_post_edit_response[i], target_new=edit_args["target_new"][i]))

        log_info.append(analyse_kl_divergence(pre_edit_logits_dict['prompt'], post_edit_logits=post_edit_logits_dict['prompt']))
        
    # Scores of the post_edit_logits
    if config.enable_output_scores:
        pre_edit_scores_string = output_token_scores(tokenizer, pre_edit_scores_dict["prompt"], 0)
        post_edit_scores_string = output_token_scores(tokenizer, post_edit_scores_dict["prompt"], 0)
        
    # Useful for debugging
    if config.enable_models_check:
        models_check_string = check_model_weights_changed(pre_edit_model,post_edit_model)
    
    
    write_output_to_file("additional_information",True,*log_info, pre_edit_scores_string, post_edit_scores_string, models_check_string)
    
    
    
    
    
    
    
    
    
    
    

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




