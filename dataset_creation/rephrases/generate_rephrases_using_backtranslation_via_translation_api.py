from deep_translator import GoogleTranslator

from utils import log, datasets_path, load_norms, get_number_of_rows, append_to_dataset




def rephrase_using_steps(questions, steps):
    
    for translator in steps:
        for i in range(len(questions)):
            questions[i] = translator.translate(questions[i])
        
    return questions




def main():

    total_number = 10
    questions = load_norms(total_number)
    
    
    steps_list = [
        
        ["en", "zh-CN", "en"],  # Chinese → English
        ["en", "ja", "en"],  # Japanese → English
        ["en", "ru", "en"],  # Russian → English  #good
        ["en", "it", "en"],  # Italian → English  #good
        ["en", "es", "en"],  # Spanish → English  #good
        ["en", "fr", "en"],  # French → English  
        ["en", "fr"],  # French → English  
        ["en", "ko", "en"],  # Korean → English  #mid
        ["en", "de", "en"],  # German → English  #godd

        ["en", "ja", "zh-CN", "en"],  # Japanese → Chinese → English
        ["en", "zh-CN", "ja", "en"],  # Chinese → Japanese → English
        ["en", "zh-CN", "it", "en"],  # Chinese → Italian → English
        ["en", "zh-CN", "ru", "en"],  # Chinese → Russian → English
    ]
    
    
    
    translator_list = [ [GoogleTranslator(source=steps[i], target=steps[i+1]) for i in range(len(steps) - 1)] for steps in steps_list]
    

    for translator_step in translator_list:
        
        results = load_norms(total_number)
        results = rephrase_using_steps(results, translator_step) 
        # append_to_dataset(questions[start_index: start_index + batch_size], results, f"./datasets/norms/rephrases/backtranslation_api/{"#".join(steps)}.json")
        print(results)
         
         
main()

