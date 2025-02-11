
from colorama import Fore, Back, Style, init
import torch
from transformers import pipeline
from datasets import load_dataset
from config import *
from utils import create_response
from utils import log, write_output_to_file
from dynamic_dataset_creation.chatgpt_utils import *






def load_norms(subset_size):
    
    ds = load_dataset("json", data_files="datasets/norms/edit_norms_dataset.json",split='train')
    # ds = ds.shuffle()
    ds = ds.select(range(subset_size))
    
    prompts = ds['prompt']
    ground_truth = ds['ground_truth']
    target_new = ds['target_new']
    subject = ds['subject']
    light_rephrase_prompts = ds['light_rephrase_prompt']
    strong_rephrase_prompts = ds['strong_rephrase_prompt']
    locality_inputs = ds['locality_inputs']
    portability_inputs = ds['portability_inputs']
    action_moral_judgment = ds["action_moral_judgment"]
    moral_action = ds["moral_action"]
    immoral_action = ds["immoral_action"]


    loc_prompts = [edit_data_['locality_inputs']['neighborhood']['prompt'] + ' ' + edit_data_['locality_inputs']['neighborhood']['ground_truth'] for edit_data_ in ds]

    # Reformat locality and probability
    locality_inputs_neighborhood_prompt_unpacked = []
    locality_inputs_neighborhood_ground_truth_unpacked = []
    locality_inputs_distracting_prompt_unpacked = []
    locality_inputs_distracting_ground_truth_unpacked = []
    
    portability_inputs_synonym_prompt_unpacked = []
    portability_inputs_synonym_ground_truth_unpacked = []
    portability_inputs_one_hop_prompt_unpacked = []
    portability_inputs_one_hop_ground_truth_unpacked = []
    
    # for l1 in locality_inputs:
    #     if len(l1['neighborhood']['prompt']) > 0:
    #         locality_inputs_neighborhood_prompt_unpacked.append(l1['neighborhood']['prompt'])
    # for l2 in locality_inputs:
    #     if len(l2['neighborhood']['ground_truth']) > 0:
    #         locality_inputs_neighborhood_ground_truth_unpacked.append(l2['neighborhood']['ground_truth'])
    # for l3 in locality_inputs:
    #     if len(l3['distracting']['prompt']) > 0:
    #         locality_inputs_distracting_prompt_unpacked.append(l3['distracting']['prompt'])
    # for l4 in locality_inputs:
    #     if len(l4['distracting']['ground_truth']) > 0:
    #         locality_inputs_distracting_ground_truth_unpacked.append(l4['distracting']['ground_truth'])
        
    # for p1 in portability_inputs:
    #     if len(p1['synonym']['prompt']) > 0:
    #         portability_inputs_synonym_prompt_unpacked.append(p1['synonym']['prompt'])
    # for p2 in portability_inputs:
    #     if len(p2['synonym']['ground_truth']) > 0:
    #         portability_inputs_synonym_ground_truth_unpacked.append(p2['synonym']['ground_truth'])
    # for p3 in portability_inputs:
    #     if len(p3['one_hop']['prompt']) > 0:
    #         portability_inputs_one_hop_prompt_unpacked.append(p3['one_hop']['prompt'])
    # for p4 in portability_inputs:
    #     if len(p4['one_hop']['ground_truth']) > 0:
    #         portability_inputs_one_hop_ground_truth_unpacked.append(p4['one_hop']['ground_truth'])
    
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

    return prompts, ground_truth, target_new, subject, light_rephrase_prompts, strong_rephrase_prompts, locality_inputs, portability_inputs, loc_prompts, action_moral_judgment, moral_action, immoral_action











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








def measure_quality_sentiment_analysis(tokenizer,edited_model, edit_args, pre_edit):
        
    # Test other metrics after editing to see whether model is degraded    
        
    negative_label = "LABEL_0"
    positive_label = "LABEL_2"
    neutral_label = "LABEL_1"


    # Compare with expected label and return float as score
    # I take a neutral label as 0.5
    def label_to_float(actual_label):
        if actual_label == neutral_label:
            return 0.5
        else:
            return float(expected_label_1 == actual_label)
    
    

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
    
    sentiment_analysis = pipeline("sentiment-analysis",model=model_name,device=0)
    
    custom_metric_array = []
    

    for index in range(0,len(edit_args["prompts"])):
        
        # To work all answers in parallel
        model_input = [edit_args["prompts"][index],
                 edited_model,tokenizer,edit_args["light_rephrase_prompts"][index],
                 edit_args["strong_rephrase_prompts"][index],
                 edit_args["moral_action"][index],
                 edit_args["immoral_action"][index]
                 ]
        
        
        # Create responses
        post_edit_output = create_response(edited_model,tokenizer,model_input,instructinoal=False)
        
        
        # Decode outputs and reformat
        decoded_post_edit_response_prompt = tokenizer.decode(post_edit_output.sequences[0],skip_special_tokens=True)
        decoded_post_edit_response_prompt = decoded_post_edit_response_prompt[len(edit_args["prompts"][index]):].lstrip(". ,")
        
        decoded_post_edit_response_light_rephrase = tokenizer.decode(post_edit_output.sequences[1],skip_special_tokens=True)
        decoded_post_edit_response_light_rephrase = decoded_post_edit_response_light_rephrase[len(edit_args["light_rephrase_prompts"][index]):].lstrip(". ,")
        
        decoded_post_edit_response_strong_rephrase = tokenizer.decode(post_edit_output.sequences[2],skip_special_tokens=True)
        decoded_post_edit_response_strong_rephrase = decoded_post_edit_response_strong_rephrase[len(edit_args["strong_rephrase_prompts"][index]):].lstrip(". ,")
        
        decoded_post_edit_response_moral_action = tokenizer.decode(post_edit_output.sequences[3],skip_special_tokens=True)
        decoded_post_edit_response_moral_action = decoded_post_edit_response_moral_action[len(edit_args["moral_action"][index]):].lstrip(". ,")
        
        decoded_post_edit_response_immoral_action = tokenizer.decode(post_edit_output.sequences[4],skip_special_tokens=True)
        decoded_post_edit_response_immoral_action = decoded_post_edit_response_immoral_action[len(edit_args["immoral_action"][index]):].lstrip(". ,")
        
        
        expected_output = edit_args["target_new"][index]
        
        if pre_edit:
            expected_output = edit_args["ground_truth"][index]


        # To work all answers in parallel
        sentiment_input = [expected_output,
                           decoded_post_edit_response_prompt,
                           decoded_post_edit_response_light_rephrase,
                           decoded_post_edit_response_strong_rephrase,
                           decoded_post_edit_response_moral_action,
                           decoded_post_edit_response_immoral_action]
        
        
        
        # Generate predictions
        sentiment_output = sentiment_analysis(sentiment_input)
        
        
        
        # Get the expected label
        # Those are predefined strings
        expected_label_1 = positive_label if int(edit_args["action_moral_judgment"][index]) > 0 else negative_label
        expected_label_2 = sentiment_output[0]["label"]
        
        
        if pre_edit:
            expected_label_1 = invert_label(expected_label_1)
        
        
        dataset_reliability = label_to_float(expected_label_2)
        
        print(f"expected_label_1: {expected_label_1}")
        print(f"expected_label_2: {expected_label_2}")
        
        
        # Test dataset reliability
        # If this fails, then every other test is almost pointless
        print(f"dataset_reliability: {dataset_reliability}")
        
        
        # Test reliability
        post_edit_label_prompt = sentiment_output[1]["label"]
        
        reliability = label_to_float(post_edit_label_prompt)
        
        print(f"decoded_post_edit_response_prompt: {decoded_post_edit_response_prompt}")
        print(sentiment_output[1])
        print(f"reliability: {reliability}")

        # Test generality
        post_edit_label_light_rephrase = sentiment_output[2]["label"]
        
        light_generality = label_to_float(post_edit_label_light_rephrase)
        
        print(f"decoded_post_edit_response_light_rephrase: {decoded_post_edit_response_light_rephrase}")
        print(sentiment_output[2])
        print(f"light_generality: {light_generality}")
        
        post_edit_label_strong_rephrase = sentiment_output[3]["label"]
        
        strong_generality = label_to_float(post_edit_label_strong_rephrase)
        
        print(f"decoded_post_edit_response_strong_rephrase: {decoded_post_edit_response_strong_rephrase}")
        print(sentiment_output[3])
        print(f"strong_generality: {strong_generality}")
        
        
        # Test one-hop inference
        post_edit_label_moral_action = sentiment_output[4]["label"]
        
        one_hop_inference = label_to_float(post_edit_label_moral_action)
        
        print(f"decoded_post_edit_response_moral_action: {decoded_post_edit_response_moral_action}")
        print(sentiment_output[4])
        print(f"one_hop_inference: {one_hop_inference}")
        
        
        # Test two-hop inference
        post_edit_label_immoral_action = sentiment_output[5]["label"]
        
        two_hop_inference = label_to_float(invert_label(post_edit_label_immoral_action))
        
        print(f"decoded_post_edit_response_immoral_action: {decoded_post_edit_response_immoral_action}")
        print(sentiment_output[5])
        print(f"two_hop_inference: {two_hop_inference}")
        
        
        custom_metric = {
            "dataset_reliability":dataset_reliability,
            "reliability":reliability,
            "light_generality":light_generality,
            "strong_generality":strong_generality,
            "one_hop_inference":one_hop_inference,
            "two_hop_inference":two_hop_inference
        }
    
        custom_metric_array.append(custom_metric)
    
    
    return custom_metric_array






def measure_quality_chatgpt_api(edit_args, decoded_post_edit_response):
    template = f"{decoded_post_edit_response}"
    response = send_request(template)











# ---------------------------------------------------------------- #
# ---------------------------------------------------------------- #
# -----------------Utils for debugging purposes------------------- #
# ---------------------------------------------------------------- #
# ---------------------------------------------------------------- #




def check_model_weights_changed(pre_edit_model, post_edit_model):
    if pre_edit_model and post_edit_model:
        output = "The models are the same."
        for parameter_name, parameter_value in pre_edit_model.state_dict().items():
            if not torch.equal(parameter_value, post_edit_model.state_dict()[parameter_name]):
                output = "The models are different."
                
        log(output,True,True,True)
        return output
    
    
    
    
    
    
def output_debugging_info(tokenizer, pre_edit_model, post_edit_model, edit_args, pre_edit_response, post_edit_response, decoded_pre_edit_response, decoded_post_edit_response):
    
    # Add log info
    log_info,pre_edit_scores_string, post_edit_scores_string,models_check_string = [] , "", "", ""
    
    # Some debugging information
    if enable_analytics:
        for i in range(len(decoded_post_edit_response)):
            log_info.append(analyse_reliability_of_edit(decoded_post_edit_response=decoded_post_edit_response[i], target_new=edit_args["target_new"][i]))

        log_info.append(analyse_kl_divergence(pre_edit_logits=pre_edit_response.logits, post_edit_logtis=post_edit_response.logits))
        
    # Scores of the post_edit_logits
    if enable_output_scores:
        pre_edit_scores_string = output_scores_of_generation(tokenizer,pre_edit_response.scores,top_k)
        post_edit_scores_string = output_scores_of_generation(tokenizer,post_edit_response.scores,top_k)
        
    # Useful for debugging
    if enable_models_check:
        models_check_string = check_model_weights_changed(pre_edit_model,post_edit_model)
        
 
    write_output_to_file("pre_edit",True,*pre_edit_scores_string)
    write_output_to_file("post_edit",True, *log_info, models_check_string, post_edit_scores_string)   
    
    
    
    
    
    
    
    
    
    
    

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




