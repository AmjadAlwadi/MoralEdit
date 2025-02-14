from colorama import Fore, Back, Style, init
from datasets import load_dataset, Dataset
import requests
import json
import time
import torch
import pandas as pd

init()



datasets_path = "./datasets"

access_token = "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"

# Set your API key
API_KEY = "002404deaad844fda84143d967491c43"
# API_KEY = "343e6b51f7cb4861a91a9b131dd289d2"

thread_id = "EYHQruY6kb9WRe4"    
# thread_id = "8GBzvkL-yYLAWgc"


# API URL
url = "https://ki-toolbox.tu-braunschweig.de/api/v1/chat/send"


# Headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    "Content-Type": "application/json",
}




available_vllm_models_for_inference = ["cognitivecomputations/dolphin-2_6-phi-2",
                    "ibm-granite/granite-3.0-2b-base",
                    "microsoft/phi-2",
                    "stabilityai/stablelm-zephyr-3b",
                    "microsoft/Phi-3.5-mini-instruct",
                    "pansophic/rocket-3B",
                    "qualcomm/Mistral-3B",
                    "teknium/OpenHermes-2.5-Mistral-7B",
                    "TheBloke/dolphin-2_6-phi-2-GPTQ",
                    "TheBloke/dolphin-2_6-phi-2-GGUF",
                    "GeneZC/MiniChat-2-3B",
                    "TheBloke/stablelm-zephyr-3b-GPTQ"
                    "TheBloke/rocket-3B-GPTQ"
                    "clibrain/mamba-2.8b-instruct-openhermes"]




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




def load_norms(subset_size, file_path=f"{datasets_path}/norms/norms_dataset.json"):
    
    ds = load_dataset("json", data_files=file_path,split='train')
    
    if subset_size != -1:
        ds = ds.select(range(subset_size))
    
    prompts = ds['rot_action']
    log(f"Norms dataset loaded with length: {len(ds)}",False,False,True)

    return prompts




def time_it(func):
    def wrapper(arg):
        start_time = time.time()  # Record start time
        result = func(arg)  # Call the function with the argument
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate time taken
        return execution_time, result
    return wrapper





def send_request(request):
    # Request payload (JSON data)
    data = {
        "thread": thread_id,   #   EYHQruY6kb9WRe4
        "prompt": request,
        "model": "gpt-4o"
    }
        
    # Send the POST request
    response = requests.post(url, headers=headers, json=data)
    # Split response into individual JSON objects
    responses = response.text.strip().split("\n")
    # print(response.text)
    result = ""
    
    # Process each JSON object
    for line in responses:
        try:
            data = json.loads(line)
            if data.get("type") == "done":
                result = data["response"]  # Print final response
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    return result



def get_number_of_rows(file_path):
    try:
        return len(load_dataset("json", data_files=file_path, split='train'))
    except:
        return 0
    
    



def append_to_dataset(prompts, rephrases, file_path):
    new_data = {"rot_action": prompts, "rephrase": rephrases}
    print("sdfsdfsdfsdfd: " + file_path)
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
    
    
    

def create_response(model,tokenizer,prompts,instructinoal:bool = False):

    model.eval()
    
    if not instructinoal:
        model_inputs = tokenizer(prompts, return_tensors='pt', padding=True, max_length=200).to(model.device)
    else:
        model_inputs = tokenizer.apply_chat_template(prompts, tokenize=True, padding=True, return_dict=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():  # Disable gradient calculations for inference
        
        outputs = model.generate(
            **model_inputs,
            num_beams = 5,
            early_stopping = True,
            do_sample = False
        )
        
    return outputs







def batch_decode_output_and_log(tokenizer,outputs,questions:str, instructional=False, output_log=False):

    decoded_outputs = tokenizer.batch_decode(outputs,skip_special_tokens=True)
    
    if instructional:
        for i in range(len(decoded_outputs)): 
            start_index = decoded_outputs[i].find("assistant\n")+ len("assistant\n")
            decoded_outputs[i] = decoded_outputs[i][start_index:]
            if output_log:
                log(decoded_output,False,True,True)
    else:
        if output_log:
            for decoded_output, question in zip(decoded_outputs, questions): 
                log(question + "-->" + decoded_output,False,True,True) 
                
                
    return decoded_outputs
    
            


def decode_output_and_log(tokenizer,output,question:str, instructional=False):

    decoded_output = tokenizer.decode(output,skip_special_tokens=True)
    
    if instructional:
        start_index = decoded_output.find("assistant\n")+ len("assistant\n")
        decoded_output = decoded_output[start_index:]
        log(decoded_output,False,True,True)
    else:
        log(question + "-->" + decoded_output,False,True,True) 

    return decoded_output



def construct_dataset(prompts, rephrases, file_path):
    data = {"rot_action":prompts, "rephrase": rephrases}
    dataset = Dataset.from_dict(data)
    dataset.to_json(file_path)