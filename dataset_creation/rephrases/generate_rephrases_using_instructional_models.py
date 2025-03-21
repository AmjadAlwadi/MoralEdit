from transformers import AutoTokenizer
from huggingface_hub import login
from transformers import set_seed
import tqdm

from utils import log, access_token, datasets_path, append_to_dataset, load_norms, create_response, batch_decode_output_and_log, decode_output_and_log, get_number_of_rows, load_norms


def load_model(model_name, access_token, text2text = False):
    
    if text2text: # Encode Decoder
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=access_token,device_map='auto')
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token,device_map='auto')
    
    log("Loaded the base model",True,False,True)
    return model




def rephrase(model, tokenizer, questions):
    
    instruction = f"Paraphrase this text"
    
    instruction_template = {"role": "system", "content": instruction}
    messages = [[instruction_template, {"role": "user", "content": question}] for question in questions]
    
    outputs = create_response(model, tokenizer, messages, True)
    questions = batch_decode_output_and_log(tokenizer, outputs, questions, True, False)
        
    return questions




def main():
    
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # "Qwen/Qwen2.5-1.5B-Instruct"
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    seed = 120
    batch_size = 20
    
    login(token=access_token,add_to_git_credential=True)
        
    if seed != -1:
        set_seed(seed)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.padding_side='left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_model(model_name, access_token, text2text=False)
    
    file_path = f"{datasets_path}/norms/rephrases/instructional_models/{model_name}.json"
    
    start_index = get_number_of_rows(file_path)
    
    questions = load_norms(-1, f"{datasets_path}/norms/norms_dataset.json")
    total_number = 200
    
    for batch_start_index in tqdm.tqdm(range(start_index, total_number, batch_size)):
    
        results = load_norms(total_number)
        results = results[batch_start_index: batch_start_index + batch_size]
        results = rephrase(model, tokenizer, results) 
        append_to_dataset(questions[batch_start_index: batch_start_index + batch_size], results, file_path)
         
         
    remaining = total_number - get_number_of_rows(file_path)
    
    if remaining > 0:
        results = load_norms(total_number)
        results = results[total_number - remaining:]
        results = rephrase(model, tokenizer, results) 
        append_to_dataset(questions[total_number - remaining:], results, file_path)
         
         
         
if __name__ == "__main__":
    main()