from chatgpt_utils import *
import os



def get_number_of_rows():
    return len(load_dataset("json", data_files="./datasets/norms/rephrases_chatgpt_api.json",split='train'))





def addToClipBoard(text):
    command = 'echo ' + text.strip() + '| clip'
    os.system(command)



def generate_rephrases(number, start_index):
    
    intervalls = 50
    
    norms = load_norms(number, file_path="./datasets/norms/norms_dataset.json")
    norms = norms[start_index:start_index + intervalls]
 
    addToClipBoard(str(norms))
    return norms

    

print(generate_rephrases(-1, get_number_of_rows()))
