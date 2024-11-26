import spacy
from datasets import load_dataset, Dataset, concatenate_datasets


nlp = spacy.load("en_core_web_sm")

def find_subject_using_speech_tagger(nlp, sentence):
    
    doc = nlp(sentence)
    
    for token in doc:
        if "SUBJ" in token.pos_:
            return token.text
    
    return None




norms = load_dataset("datasets/norms/", data_files="norms_dataset.json", split='train')


very_good_words = load_dataset("datasets/judgements/", data_files="very_good_dataset.json", split='train')
good_words = load_dataset("datasets/judgements/", data_files="good_dataset.json", split='train')
bad_words = load_dataset("datasets/judgements/", data_files="bad_dataset.json", split='train')
very_bad_words = load_dataset("datasets/judgements/", data_files="very_bad_dataset.json", split='train')
ok_words = load_dataset("datasets/judgements/", data_files="ok_dataset.json", split='train')

# Datasets for judgements to check later if no match was found in the first check
big_list_bad_words = load_dataset("datasets/judgements/", data_files="bad_and_very_bad_dataset.json", split='train')
big_list_good_words = load_dataset("datasets/judgements/", data_files="good_and_very_good_dataset.json", split='train')


adjectives_dataset = concatenate_datasets([very_good_words,good_words,bad_words,very_bad_words,ok_words,big_list_bad_words,big_list_good_words])


edit_norms_size = len(norms)
norms_subset = norms.select(range(edit_norms_size))

def find_subject(batch, indices):
    new_items = []
    
    prepositions = [
    "about", "above", "across", "after", "against", "along", "among", 
    "around", "at", "before", "behind", "below", "beneath", "beside", 
    "between", "by", "down", "during", "for", "from", "in", "inside", 
    "into", "near", "of", "off", "on", "over", "through", "to", "under", 
    "up", "with", "without"
    ]

    # use a very big dataset of adjectives and verbs


    words_to_skip = ["my","your","his","her","it's","their",
                     "a","an","the","and","or",
                     "and", "but", "or", "nor", "for", "so", "yet",
                     "although","though","since", "unless", "while",
                     "neither", "both", "only", "also",
                     "as", "until", "till","other"

                    # Conjunctions
                    "and", "or", "but", "nor", "for", "so", "yet",
                    
                    # Auxiliary Verbs
                    "am", "is", "are", "was", "were", "be", "being", "been", "have", 
                    "has", "had", "do", "does", "did", "will", "would", "shall", 
                    "should", "can", "could", "may", "might", "must",
                    
                    # Interjections (for completeness)
                    "ah", "oh", "hey", "ouch", "wow",
                    "back","forth",
                    
                    "what", "where", "when", "how", "which", 
                    "never", "away"
                ]


    
    for i,idx in enumerate(indices):
        
        subject = ""
        sentence = batch['rot-action'][i].lower()
        sentence = sentence.rstrip("?")
        sentence = sentence.rstrip(".")
        sentence = sentence.split(" ")
        length = len(sentence)
        
        for i, word in enumerate(sentence):
            
            if len(word) == 0:
                continue
            
            
            # Skip prepositions and words to skip
            if word in prepositions or word in words_to_skip:
                continue
            
            # Skip verbs
            if word.endswith("ing"):
                if i == length - 1:
                    subject = word
                continue
             
             
            # Skip adjectives and adverbs     
            for adjective in adjectives_dataset['judgement']:
                if word == adjective.lower() or word == adjective.lower() + 'ly' or word == adjective.lower() + 'er' or word == adjective.lower() + 'est':
                    continue
            
            
            subject = word
            break
            
            
        new_items.append(subject)
                
    # return batch
    return {"subject": new_items}


# Convert the result into a new dataset
result = norms_subset.map(find_subject, with_indices=True, batched=True, batch_size=3000)

new_items_list = [item for item in result["subject"]]
new_items_dict = {"subject": new_items_list}

subejcts_dataset = Dataset.from_dict(new_items_dict)
subejcts_dataset.to_json("datasets/norms/subjects_1.json")

