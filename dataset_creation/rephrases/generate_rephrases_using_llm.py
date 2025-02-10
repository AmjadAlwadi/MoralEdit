from vllm import LLM, SamplingParams
from colorama import Fore, Back, Style, init
from datasets import load_dataset, Dataset
import argparse
import warnings 

from utils import log, datasets_path, construct_dataset, load_norms


# Ignore all warnings
warnings.filterwarnings("ignore")


def generate(number_of_norms):
    
    model = "pansophic/rocket-3B"
    instruction ="Rephrase this very slightly!"
    instruction_2 ="reword a single word and only give me the result back without any explanations!"
    instruction_3 ="Find the subject in this sentence!"
    
    rephrases = []
    prompts = []
    norms = load_norms(number_of_norms)
    
    system = instruction
    
    for norm in norms:
        
        user = norm + '.'
        prompt = f"""<|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        {user}<|im_end|>
        <|im_start|>assistant
        """
        prompts.append(prompt)
    
    
        
    llm = LLM(model=model,dtype='half',enforce_eager=True, max_model_len=700, gpu_memory_utilization=0.999)


    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=28)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        raw_output = output.outputs[0].text
        result = ""
        continuation = " is"
        
        if raw_output.find('?') == -1:
            result = raw_output.split('.')
        else:
            result = raw_output.split('?')
            continuation = '?'
            
        rephrases.append(result[0] + continuation)
        
    return norms, rephrases    
    
        
        
        
        

if __name__ == "__main__":
    argparse.ArgumentParser = argparse.ArgumentParser()
    argparse.ArgumentParser.add_argument("-n","--number_of_norms", type=int, default=-1, help="Number of norms to generate")
    args = argparse.ArgumentParser.parse_args()
    
    construct_dataset(*generate(args.number_of_norms), "./datasets/norms/rephrases/vllm/rephrases_vllm.json")