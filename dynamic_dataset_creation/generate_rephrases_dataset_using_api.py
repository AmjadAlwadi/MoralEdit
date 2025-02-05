from chatgpt_utils import *


@time_it 
def generate_rephrase(prompt):
    return send_request(f"Rephrase the following: {prompt}")



def generate_rephrases(number):
    
    time_needed = 0
    norms = load_norms(number)
    
    for i, prompt in enumerate(norms, start=1):
        time_for_one, rephrase = generate_rephrase(prompt)
        log(f"{prompt} -> {rephrase}",False,False,False)
        time_needed += time_for_one
    
    log(f"Total time taken: {time_needed:.2f} seconds",True,False,True)
    

generate_rephrases(10)
