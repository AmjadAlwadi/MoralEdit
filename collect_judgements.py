from datasets import load_dataset,Dataset

dataset_size = 100
# dataset_size = 9500

# Load the datasets
social_chemistry = load_dataset("datasets/social-chem-101/", data_files="social-chem-101.v1.0.tsv", split='train')
social_chemistry = social_chemistry.remove_columns(['split','area', 'm', 'rot-agree', 'rot-categorization', 'rot-moral-foundations', 'rot-char-targeting', 'rot-bad', 'action', 'action-agency', 'action-agree', 'action-legal', 'action-pressure', 'action-char-involved', 'action-hypothetical', 'situation', 'situation-short-id', 'rot-worker-id', 'breakdown-worker-id', 'n-characters', 'characters'])

# Create an empty dataset
very_bad_dataset = Dataset.from_dict({"judgement":["bad"]})
bad_dataset = Dataset.from_dict({"judgement":["very bad"]})
ok_dataset = Dataset.from_dict({"judgement":["ok"]})
good_dataset = Dataset.from_dict({"judgement":["good"]})
very_good_dataset = Dataset.from_dict({"judgement":["very good"]})


print(social_chemistry)


def add_judgements(example,index):
    global very_bad_dataset,bad_dataset,ok_dataset,good_dataset,very_good_dataset
    
    for element_1, element_2 in zip(example["rot-judgment"],example["action-moral-judgment"]):
        if element_2 == -2:
            very_bad_dataset = very_bad_dataset.add_item({"judgement":element_1})
        elif element_2 == -1:
            bad_dataset = bad_dataset.add_item({"judgement":element_1})
        elif element_2 == 0:
            ok_dataset = ok_dataset.add_item({"judgement":element_1})
        elif element_2 == 1:
            good_dataset = good_dataset.add_item({"judgement":element_1})
        elif element_2 == 2:
            very_good_dataset = very_good_dataset.add_item({"judgement":element_1})
        else:
            return example
    
    return example


# Apply the adjust function to the dataset
social_chemistry = social_chemistry.map(add_judgements,with_indices=True, batched=True, batch_size=3000)

very_bad_dataset.to_json("very_bad_dataset.json")
bad_dataset.to_json("bad_dataset.json")
ok_dataset.to_json("ok_dataset.json")
good_dataset.to_json("good_dataset.json")
very_good_dataset.to_json("very_good_dataset.json")
