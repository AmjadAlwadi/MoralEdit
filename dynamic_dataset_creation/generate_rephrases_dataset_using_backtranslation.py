from datasets import load_dataset, Dataset       
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from colorama import Fore, Back, Style, init

from huggingface_hub import login
from transformers import set_seed


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




def create_response(model,tokenizer,prompts,instructinoal:bool = False):

    model.eval()
    
    if not instructinoal:
        model_inputs = tokenizer(prompts, return_tensors='pt', padding=True, max_length=200).to(model.device)
    else:
        model_inputs = tokenizer.apply_chat_template(prompts, tokenize=True, padding=True, return_dict=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():  # Disable gradient calculations for inference
        
        outputs = model.generate(
            **model_inputs,
            num_beams = 3,
            early_stopping = True,
            do_sample = False
        )
        
    return outputs





def decode_output_and_log(tokenizer,output,question:str, instructional=False):

    decoded_output = tokenizer.decode(output,skip_special_tokens=True)
    
    if instructional:
        start_index = decoded_output.find("assistant\n")+ len("assistant\n")
        decoded_output = decoded_output[start_index:]
        log(decoded_output,False,True,True)

    else:
        log('Outputs: ' + question + Back.LIGHTBLACK_EX + decoded_output[len(question):],False,True,True)  

    return decoded_output




def load_model(model_name, access_token, text2text = False):
    
    if text2text: # Encode Decoder
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=access_token,device_map='auto')
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token,device_map='auto')
    
    log("Loaded the base model",True,False,True)
    return model





def rephrase_using_steps(model, tokenizer, questions):
    
    # steps =  ["Japanese", "Chinese", "English"]

    steps =  ["Chinese", "English"]
    
    # steps =  ["Spanish", "Russian", "English"]
    steps =  ["Spanish", "French", "English"]
    

    for step in steps:
        instruction = f"Translate this text into {step}"
        
        instruction_template = {"role": "system", "content": instruction}
        messages = [[instruction_template, {"role": "user", "content": question}] for question in questions]
        
        output = create_response(model, tokenizer, messages, True)
        
        for i in range(len(questions)):
            questions[i] = decode_output_and_log(tokenizer, output[i], questions[i], True)
        
    return questions





def load_norms(subset_size = -1, file_path="./datasets/norms/norms_dataset.json"):
    
    ds = load_dataset("json", data_files=file_path,split='train')
    
    if subset_size != -1:
        ds = ds.select(range(subset_size))
    
    prompts = ds['rot_action']
    log(f"Norms dataset loaded with length: {len(ds)}",False,False,True)

    return prompts





def main():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    access_token = "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"
    seed = 120

    login(token=access_token,add_to_git_credential=True)
        
    if seed != -1:
        set_seed(seed)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.padding_side='left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    
    model = load_model(model_name, access_token, text2text=False)
    
    number = 5
    
    questions = load_norms(number)
    results = load_norms(number)
    results = rephrase_using_steps(model, tokenizer, results)
    
    results_dict = {"rot_action": questions, "rephrases":results}

    dataset = Dataset.from_dict(results_dict)
    dataset.to_json("./datasets/norms/rephrases_backtranslation.json")
    
    
main()