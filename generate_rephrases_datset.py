from vllm import LLM, SamplingParams
from colorama import Fore, Back, Style, init
from datasets import load_dataset, Dataset
import argparse
import warnings 

# Ignore all warnings
warnings.filterwarnings("ignore")

init()

available_models_for_inference = ["cognitivecomputations/dolphin-2_6-phi-2",
                    "ibm-granite/granite-3.0-2b-base",
                    "microsoft/phi-2",
                    "stabilityai/stablelm-zephyr-3b",
                    "microsoft/Phi-3.5-mini-instruct",
                    "pansophic/rocket-3B",
                    "qualcomm/Mistral-3B",
                    "teknium/OpenHermes-2.5-Mistral-7B",
                    "TheBloke/dolphin-2_6-phi-2-GPTQ",
                    "TheBloke/dolphin-2_6-phi-2-GGUF",
                    "GeneZC/MiniChat-2-3B",
                    "TheBloke/stablelm-zephyr-3b-GPTQ"
                    "TheBloke/rocket-3B-GPTQ"
                    "clibrain/mamba-2.8b-instruct-openhermes"]


def load_norms(subset_size):
    
    ds = load_dataset("json", data_files="datasets/norms/norms_dataset.json",split='train')
    if subset_size != -1:
        ds = ds.select(range(subset_size))
    
    prompts = ds['rot-action']
    log(f"Norms dataset loaded with length: {len(ds)}",False,False,True)

    return prompts





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
    
        
        
def construct_dataset(prompts, rephrases):
    data = {"prompts":prompts, "rephrase": rephrases}
    dataset = Dataset.from_dict(data)
    dataset.to_json("datasets/norms/rephrases.json")
        
        

if __name__ == "__main__":
    argparse.ArgumentParser = argparse.ArgumentParser()
    argparse.ArgumentParser.add_argument("-n","--number_of_norms", type=int, default=-1, help="Number of norms to generate")
    args = argparse.ArgumentParser.parse_args()
    
    construct_dataset(*generate(args.number_of_norms))