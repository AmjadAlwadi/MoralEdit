import spacy
from datasets import load_dataset, Dataset, concatenate_datasets

mismatch_string = "!?:)(:?!"

nlp = spacy.load("en_core_web_sm")


def swap_aux_and_subj(sentence):
    if sentence[-1] == '.':
        sentence = sentence[:-1]
    elif sentence[-1] == '?':
        print("Can't rephrase this.")
        return sentence
    
    doc = nlp(sentence)
    
    aux = None
    subj = None
    
    # Find the first aux and subj
    for token in doc:
        if aux is None and "aux" in token.dep_:
            aux = token
        if subj is None and "subj" in token.dep_:
            subj = token
        if aux and subj:
            break
    
    if aux and subj:
        
        if aux == "'s":
            aux = "is"
        elif aux == "re":
            aux = "are"
        elif aux == "ve":
            aux = "have"
        
        tokens = [token.text for token in doc]
        
        tokens[aux.i], tokens[subj.i] = tokens[subj.i].lower(), tokens[aux.i].capitalize()
        
        new_sentence = " ".join(tokens) + '?'
        return new_sentence
    
    return sentence




def debug(sentence):
    doc = nlp(sentence)
    
    for token in doc:
        print(token.text, token.dep_)



norms = load_dataset("../datasets/norms/", data_files="norms_dataset.json", split='train')

edit_norms_size = 100
norms_subset = norms.select(range(edit_norms_size))


def rephrase(batch, indices):
    new_items = []
    
    for i,idx in enumerate(indices):
        
        sentence = batch['original_norm'][i]   
        rephrase = swap_aux_and_subj(sentence)
        new_items.append(rephrase)
                
    # return batch
    return {"rephrase": new_items}


# Convert the result into a new dataset
result = norms_subset.map(rephrase, with_indices=True, batched=True, batch_size=3000)

new_items_list = [item for item in result["rephrase"]]
new_items_dict = {"rephrase": new_items_list}

rephrases_dataset = Dataset.from_dict(new_items_dict)
rephrases_dataset.to_json("../datasets/norms/rephrases_st.json")

