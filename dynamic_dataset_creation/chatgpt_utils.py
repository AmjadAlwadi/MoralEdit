from colorama import Fore, Back, Style, init
from datasets import load_dataset, Dataset
import requests
import json
import time

init()


# ---------------------------------------------------------------- #
# ---------------------------------------------------------------- #
# ---------------Helper functions for ChatGPT API----------------- #
# ---------------------------------------------------------------- #
# ---------------------------------------------------------------- #


# Set your API key
# API_KEY = "002404deaad844fda84143d967491c43"
API_KEY = "343e6b51f7cb4861a91a9b131dd289d2"




# API URL
url = "https://ki-toolbox.tu-braunschweig.de/api/v1/chat/send"


# Headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    "Content-Type": "application/json",
}



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




def load_norms(subset_size):
    
    ds = load_dataset("json", data_files="../datasets/norms/norms_dataset.json",split='train')
    
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
        "thread": "8GBzvkL-yYLAWgc",   #   EYHQruY6kb9WRe4
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



