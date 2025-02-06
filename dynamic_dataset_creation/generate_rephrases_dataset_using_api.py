from chatgpt_utils import *
import argparse
import tqdm
import pandas as pd


@time_it 
def generate_rephrase(prompt):
    return send_request(f"Rephrase the following: {prompt}")




def get_number_of_rows():
    return len(load_dataset("json", data_files="../datasets/norms/rephrases_chatgpt_api.json",split='train'))






def generate_rephrases(number, start_index):
    
    intervalls = 30
    
    time_needed = 0
    norms = load_norms(number)
    norms = norms[start_index - 1:]
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
    
    
    
    
    

def append_to_dataset(prompts, rephrases, file_path="../datasets/norms/rephrases_chatgpt_api.json"):
    new_data = {"rot_action": prompts, "rephrase": rephrases}
    
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




        
        

if __name__ == "__main__":
    argparse.ArgumentParser = argparse.ArgumentParser()
    argparse.ArgumentParser.add_argument("-n","--number_of_norms", type=int, default=-1, help="Number of norms to generate")
    argparse.ArgumentParser.add_argument("-s","--start_index", type=int, default=get_number_of_rows() + 1, help="Where to start")

    args = argparse.ArgumentParser.parse_args()
    
    generate_rephrases(args.number_of_norms, args.start_index)