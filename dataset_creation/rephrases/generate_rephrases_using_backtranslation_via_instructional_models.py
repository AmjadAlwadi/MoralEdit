from transformers import AutoTokenizer
from colorama import Fore, Back, Style, init
from huggingface_hub import login
from transformers import set_seed

from utils import access_token, log, datasets_path, load_norms, get_number_of_rows, append_to_dataset, create_response, decode_output_and_log, batch_decode_output_and_log



def load_model(model_name, access_token, text2text = False):
    
    if text2text: # Encode Decoder
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=access_token,device_map='auto')
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token,device_map='auto')
    
    log("Loaded the base model",True,False,True)
    return model




def rephrase_using_steps(model, tokenizer, questions, steps):
    
    for step in steps:
        instruction = f"Translate this text into {step}"
        
        instruction_template = {"role": "system", "content": instruction}
        messages = [[instruction_template, {"role": "user", "content": question}] for question in questions]
        
        output = create_response(model, tokenizer, messages, True)
        
        for i in range(len(questions)):
            questions[i] = decode_output_and_log(tokenizer, output[i], questions[i], True)
        
    return questions




def main():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    seed = 120

    login(token=access_token,add_to_git_credential=True)
        
    if seed != -1:
        set_seed(seed)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.padding_side='left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    
    model = load_model(model_name, access_token, text2text=False)
    
    total_number = 100
    batch_size = 20
    
    
    questions = load_norms(total_number)
    # norms = norms[get_number_of_rows(f"{datasets_path}/norms/rephrases_backtranslation_Chinese#English.json"):]
    
    # Try to paraphrase
    # Translate using translator not models
    
    # ["Chinese", "English"],  #0  
    # ["Japanese", "English"], #1.5  #8 #3.5
    # ["Russian", "English"], #1.5
    # ["Italian", "English"],  #2 #2 #1
    # ["Spanish", "English"], #2 #2
    # ["French", "English"],  #2  #2
    # ["Korean", "English"], #5  #2.5
    # ["German", "English"], #4  #3
    
    # ["Japanese", "Chinese", "English"],
    # ["Chinese", "Japanese", "English"], # 1.5
    
    # ["Chinese", "Italian", "English"],
    # ["Chinese", "Russian", "English"], # 1.5

    
    steps = ["Chinese", "English"]
    
    
    for start_index in range(0, total_number, batch_size):
        results = load_norms(total_number)
        results = results[start_index: start_index + batch_size]
        results = rephrase_using_steps(model, tokenizer, results, steps) 
        append_to_dataset(questions[start_index: start_index + batch_size], results, f"{datasets_path}/norms/rephrases/backtranslation_instructional_models/{"#".join(steps)}.json")
         
         
         
main()