import spacy
from datasets import load_dataset, Dataset, concatenate_datasets

nlp = spacy.load("en_core_web_sm")

match_priority_list = ["subj","root","obj"]
mismatch_string = "!?:)(:?!"



def find_match_using_speech_tagger(sentence):
    doc = nlp(sentence)
    
    for priority in match_priority_list:
        for token in doc:
            if priority in token.dep_.lower():
                return token.text
    return mismatch_string



def debug(sentence):
    doc = nlp(sentence)
    
    for token in doc:
        print(token.text, token.dep_)





norms = load_dataset("../datasets/norms/", data_files="norms_dataset.json", split='train')

edit_norms_size = len(norms)
norms_subset = norms.select(range(edit_norms_size))


def find_subject(batch, indices):
    new_items = []
    
    for i,idx in enumerate(indices):
        
        sentence = batch['rot-action'][i]   
        subject = find_match_using_speech_tagger(sentence)
        new_items.append(subject)
                
    # return batch
    return {"subject": new_items}


# Convert the result into a new dataset
result = norms_subset.map(find_subject, with_indices=True, batched=True, batch_size=3000)

new_items_list = [item for item in result["subject"]]
new_items_dict = {"subject": new_items_list}

subejcts_dataset = Dataset.from_dict(new_items_dict)
subejcts_dataset.to_json("../datasets/norms/subjects_st.json")

