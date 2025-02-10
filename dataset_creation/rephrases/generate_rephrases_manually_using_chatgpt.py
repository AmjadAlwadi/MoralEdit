import os
from datasets import load_dataset
from utils import get_number_of_rows, load_norms




def get_number_of_rows(file_path):
    try:
        return len(load_dataset("json", data_files=file_path, split='train'))
    except:
        return 0
    

def addToClipBoard(text):
    command = 'echo ' + text.strip() + '| clip'
    os.system(command)



def generate_rephrases(number, start_index):
    
    intervalls = 50
    
    norms = load_norms(number, file_path="./datasets/norms/norms_dataset.json")
    norms = norms[start_index:start_index + intervalls]
 
    addToClipBoard(str(norms))
    return norms

    

print(generate_rephrases(-1, get_number_of_rows("./datasets/norms/rephrases/gpt4o_api/rephrases.json")))
