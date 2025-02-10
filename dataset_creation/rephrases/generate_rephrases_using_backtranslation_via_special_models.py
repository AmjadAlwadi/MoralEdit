from transformers import AutoTokenizer, MarianMTModel
from huggingface_hub import login
from transformers import set_seed
import tqdm

from utils import access_token, datasets_path, load_norms, get_number_of_rows, append_to_dataset, create_response, decode_output_and_log, batch_decode_output_and_log




def rephrase_using_steps(models, toks, questions):
    
    for i in range(len(models)):
        output = create_response(models[i], toks[i], questions)
        
        for j in range(len(questions)):
            questions[j] = decode_output_and_log(toks[i], output[j], questions[j])
    return questions





def batch_rephrase_using_steps(models, toks, questions):
    
    for i in range(len(models)):
        outputs = create_response(models[i], toks[i], questions)
        questions = batch_decode_output_and_log(toks[i], outputs, questions, True)
        
    return questions




def main():
    
    seed = 120

    login(token=access_token,add_to_git_credential=True)
        
    if seed != -1:
        set_seed(seed)

    batch_size = 10
    
    steps = ["en", "zh", "en"]

    toks = []
    models = []
    
    file_path = f"{datasets_path}/norms/rephrases/backtranslation_special_models/{"#".join(steps)}.json"
    
    start_index = get_number_of_rows(file_path)
    
    questions = load_norms(-1, f"{datasets_path}/norms/norms_dataset.json")
    total_number = len(questions)
    
    
    #["en", "zh", "en"],  # Chinese → English
    #["en", "ja", "en"],  # Japanese → English
    #["en", "ru", "en"],  # Russian → English  #good
    #["en", "it", "en"],  # Italian → English  #good
    #["en", "es", "en"],  # Spanish → English  #good
    #["en", "fr", "en"],  # French → English  
    #["en", "fr"],  # French → English  
    #["en", "ko", "en"],  # Korean → English  #mid
    #["en", "de", "en"],  # German → English  #godd

    #["en", "ja", "zh", "en"],  # Japanese → Chinese → English
    #["en", "zh", "ja", "en"],  # Chinese → Japanese → English
    #["en", "zh", "it", "en"],  # Chinese → Italian → English
    #["en", "zh", "ru", "en"],  # Chinese → Russian → English
    
    
    
    for i in range(len(steps) - 1):
        src_lang = steps[i]
        tgt_lang = steps[i + 1]
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        toks.append(tokenizer)
        models.append(model)
    

    
    for batch_start_index in tqdm.tqdm(range(start_index, total_number, batch_size)):
        
        results = load_norms(-1, f"{datasets_path}/norms/norms_dataset.json")
        results = results[batch_start_index: batch_start_index + batch_size]
        results = batch_rephrase_using_steps(models, toks, results) 
        append_to_dataset(questions[batch_start_index: batch_start_index + batch_size], results, file_path)
        

    remaining = total_number - get_number_of_rows(file_path)
    
    if remaining > 0:
        results = load_norms(-1, f"{datasets_path}/norms/norms_dataset.json")
        results = results[total_number - remaining:]
        results = batch_rephrase_using_steps(models, toks, results)
        append_to_dataset(questions[total_number - remaining:], results, file_path)
    
    

if __name__ == '__main__':
    main()
