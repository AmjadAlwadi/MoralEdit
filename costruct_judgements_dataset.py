from datasets import load_dataset,Dataset


good_words_dataset = load_dataset("csv",data_files="datasets/judgements/good_words.csv", split='train')
good_words_dataset_1 = load_dataset("json",data_files="datasets/judgements/good_dataset.json", split='train')
good_words_dataset_2 = load_dataset("json",data_files="datasets/judgements/very_good_dataset.json", split='train')
# ok_dataset = load_dataset("json",data_files="datasets/judgements/ok_dataset.json", split='train')
bad_words_dataset = load_dataset("csv",data_files="datasets/judgements/bad_words.csv", split='train')
bad_words_dataset_1 = load_dataset("json",data_files="datasets/judgements/bad_dataset.json", split='train')
bad_words_dataset_2 = load_dataset("json",data_files="datasets/judgements/very_bad_dataset.json", split='train')


good_words = set()
bad_words = set()


def remove_duplicates_good(examples):
    global good_words
    for example in examples['judgement']:
        if example not in good_words:
            good_words.add(example)
    return examples


def remove_duplicates_bad(examples):
    global bad_words
    for example in examples['judgement']:
        if example not in bad_words:
            bad_words.add(example)
    return examples


def lower_case(examples):
    examples['judgement'] = examples['judgement'].lower()
    return examples




good_words_dataset = good_words_dataset.map(lower_case)
bad_words_dataset = bad_words_dataset.map(lower_case)
good_words_dataset_1 = good_words_dataset_1.map(lower_case)
good_words_dataset_2 = good_words_dataset_2.map(lower_case)
bad_words_dataset_1 = bad_words_dataset_1.map(lower_case)
bad_words_dataset_2 = bad_words_dataset_2.map(lower_case)



good_words_dataset = good_words_dataset.map(remove_duplicates_good, batched=True,batch_size=100,load_from_cache_file=False)
bad_words_dataset = bad_words_dataset.map(remove_duplicates_good, batched=True,batch_size=100,load_from_cache_file=False)
good_words_dataset_1 = good_words_dataset_1.map(remove_duplicates_good, batched=True,batch_size=100,load_from_cache_file=False)
good_words_dataset_2 = good_words_dataset_2.map(remove_duplicates_good, batched=True,batch_size=100,load_from_cache_file=False)
bad_words_dataset_1 = bad_words_dataset_1.map(remove_duplicates_bad, batched=True,batch_size=100,load_from_cache_file=False)
bad_words_dataset_2 = bad_words_dataset_2.map(remove_duplicates_bad, batched=True,batch_size=100,load_from_cache_file=False)



good_words_dict = {'judgement': good_words}
bad_words_dict = {'judgement': bad_words}


good_and_very_good_dataset = Dataset.from_dict(good_words_dict)
bad_and_very_bad_dataset = Dataset.from_dict(bad_words_dict)

good_and_very_good_dataset.to_json("datasets/judgements/good_and_very_good_dataset.json")
bad_and_very_bad_dataset.to_json("datasets/judgements/bad_and_very_bad_dataset.json")