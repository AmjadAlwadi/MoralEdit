from utils import *
import tqdm


@time_it 
def generate_rephrase(prompt):
    return send_request(f"Rephrase the following: {prompt}")



def generate_rephrases(number, start_index):
    
    intervalls = 20
    
    time_needed = 0
    norms = load_norms(number, file_path=f"{datasets_path}/norms/norms_dataset.json")
    norms = norms[start_index:]
    rephrases = []
    
    if len(norms) == 0:
        print("Done")
    
    for i, prompt in tqdm.tqdm(enumerate(norms, start=1)):
        time_for_one, rephrase = generate_rephrase(prompt)
        
        if len(rephrase) == 0:
            print(f"Generation stopped")
            return
        
        log(f"{prompt} -> {rephrase}",False,False,False)
        time_needed += time_for_one
        rephrases.append(rephrase)
        
        if len(rephrases) == intervalls:
            
            append_to_dataset(norms[:intervalls], rephrases[:intervalls])
            
            norms = norms[intervalls:]
            rephrases = rephrases[intervalls:]
    
    log(f"Total time taken: {time_needed:.2f} seconds",True,False,True)
    return norms, rephrases
    
    
    
if __name__ == "__main__": 
    generate_rephrases(-1, get_number_of_rows(f"{datasets_path}/norms/rephrases/gpt4o_api/rephrases.json"))