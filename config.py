from datetime import datetime
from utils import get_datasets_path
from utils import count_directories, get_ml_path
import os

# Global constants
timestamp = datetime.now().strftime("%d-%m-%Y__%H-%M")


access_token = "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"

available_editing_methods = { 0: "ROME", 1: "R-ROME", 2: "MEMIT", 3: "EMMET", 4: "PMET", 5: "IKE", 6: "GRACE", 7: "MELO", 8: "WISE", 9: "DPO", 10: "INSTRUCTION_ENGINEERING", # Do not require pretraining
                             11: "FT-L", 12: "FT-M", 13: "LORA", 14: "QLORA",
                             15: "MEND", 16: "SERAC", 17: "MALMEN"}

available_models = {
    0: "meta-llama/Llama-2-7b", #FP32
    1: "meta-llama/Llama-2-7b-hf", #FP16
    2: "meta-llama/Meta-Llama-3-8B", 3: "meta-llama/Meta-Llama-3-8B-Instruct", #BF16
    4: "meta-llama/Llama-3.1-8B", 5: "meta-llama/Llama-3.1-8B-Instruct", #BF16
    6: "meta-llama/Llama-3.2-1B", 7: "meta-llama/Llama-3.2-1B-Instruct", #BF16
    8: "meta-llama/Llama-3.2-3B", 9: "meta-llama/Llama-3.2-3B-Instruct", #BF16

    10: "openai-community/gpt2-xl", 11: "EleutherAI/gpt-j-6b", 12: "EleutherAI/gpt-neo-2.7B", #F32
    
    13: "Qwen/Qwen-1_8B", 14: "Qwen/Qwen-7B-Chat", #BF16
    15: "Qwen/Qwen1.5-0.5B", 16: "Qwen/Qwen1.5-0.5B-Chat", #BF16
    17: "Qwen/Qwen1.5-1.8B", 18: "Qwen/Qwen1.5-4B", #BF16
    19: "Qwen/Qwen1.5-4B-Chat", 20: "Qwen/Qwen1.5-7B", #BF16
    21: "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8", #BF16
    
    22: "Qwen/Qwen2-0.5B", 23: "Qwen/Qwen2-0.5B-Instruct", #BF16
    24: "Qwen/Qwen2-1.5B", 25: "Qwen/Qwen2-1.5B-Instruct", #BF16
    26: "Qwen/Qwen2-7B", 27: "Qwen/Qwen2-7B-Instruct", #BF16
    28: "Qwen/Qwen2.5-0.5B", 29: "Qwen/Qwen2.5-0.5B-Instruct", #BF16
    30: "Qwen/Qwen2.5-1.5B", 31: "Qwen/Qwen2.5-1.5B-Instruct", #BF16
    32: "Qwen/Qwen2.5-3B", 33: "Qwen/Qwen2.5-3B-Instruct", #BF16
    
    34: "mistralai/Mistral-7B-v0.1", 35: "mistralai/Mistral-7B-Instruct-v0.2", 36: "mistralai/Mistral-7B-v0.3", 37: "qualcomm/Mistral-3B", #BF16
    
    38: "google-t5/t5-3b" #F32
}



# For vllm
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



# Global config variables
datasets_path = get_datasets_path()


# General configuration
model_name = "openai-community/gpt2-xl"
device = 0
train = False
weights_dtype = None
norms_subset_size = 1
shuffle = True

# For decoding strategy
decoding_strategy = "greedy decoding"
do_sample = True
num_beams = 1
max_length = 100
no_repeat_ngram_size = 0
early_stopping = True
max_new_tokens = 8
num_return_sequences = num_beams
seed = -1
# temperature = 1e-10  # To ensure reproducibility
# top_k = 50
# top_p = 0.95


# For editing
editing_method = "MEND"
hparams_path = None
train_hparams_path = None
ike_loc_examples_number = 2


# Evaluation
enable_sentiment = False
enable_kl_div = False
enable_perplexity = False
batching = False
norms_dataset_number = 0


# For debugging
enable_models_check = False
enable_analytics = False
enable_output_scores = False
scores_top_k = 5
freely_chat_with_post_edit_model = False
num_dirs = 0


