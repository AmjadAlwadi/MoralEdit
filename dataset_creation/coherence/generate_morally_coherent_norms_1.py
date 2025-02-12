from datasets import load_dataset
from transformers import pipeline
import torch
import argparse

datasets_path = "./datasets"


def load_norms(subset_size, shuffle):
    norms = load_dataset(f"{datasets_path}/norms/", data_files="norms_dataset.json", split='train')
    n = len(norms) if subset_size == -1 else subset_size
    if shuffle:
        norms = norms.shuffle()
    norms_subset = norms.select(range(n))
    return norms_subset





def load_edit_norms(subset_size, shuffle):
    edit_norms = load_dataset(f"{datasets_path}/norms/edit_norms_datasets/", data_files="edit_norms_dataset.json", split='train')
    n = len(edit_norms) if subset_size == -1 else subset_size
    if shuffle:
        edit_norms = edit_norms.shuffle()
    edit_norms_subset = edit_norms.select(range(n))
    return edit_norms_subset




def is_textually_entailed_batch(texts, hypotheses, classifier, entailment_threshold, batch_size):
    inputs = [f"{text.rstrip('.?')}. {hypothesis.rstrip('.?')}." for text, hypothesis in zip(texts, hypotheses)]
    results = classifier(inputs, batch_size=batch_size)
    entailments = [
        result['label'] == 'ENTAILMENT' and result['score'] >= entailment_threshold
        for result in results
    ]
    return entailments



def is_textually_contradicted_batch(texts, hypotheses, classifier, contradiction_threshold, batch_size):
    inputs = [f"{text.rstrip('.?')}. {hypothesis.rstrip('.?')}." for text, hypothesis in zip(texts, hypotheses)]
    results = classifier(inputs, batch_size=batch_size)
    contradictions = [
        result['label'] == 'CONTRADICTION' and result['score'] >= contradiction_threshold
        for result in results
    ]
    return contradictions





def filter_moral_batch(batch, classifier, entailment_threshold, batch_size):
    situation_moral_actions = batch['moral_action']
    rot_actions = batch['prompt']
    
    for i in range(len(rot_actions)):
        situation_moral_actions[i] = f"{batch['situation'][i]} {batch['moral_action'][i][:-15]}."
        rot_actions[i] = rot_actions[i][:-3]
    
    entailments = is_textually_entailed_batch(situation_moral_actions, rot_actions, classifier, entailment_threshold, batch_size)
    return entailments






def filter_immoral_batch(batch, classifier, contradiction_threshold, batch_size):

    situation_immoral_actions = batch['immoral_action']
    rot_actions = batch['prompt']
    
    for i in range(len(rot_actions)):
        situation_immoral_actions[i] = f"{batch['situation'][i]} {batch['immoral_action'][i][:-15]}."
        rot_actions[i] = rot_actions[i][:-3]
    
    contradictions = is_textually_contradicted_batch(situation_immoral_actions, rot_actions, classifier, contradiction_threshold, batch_size)
    return contradictions





if __name__ == '__main__':

    datasets_path = "../../datasets"
    
    device = torch.device('cuda')
    classifier = pipeline("text-classification", model = "roberta-large-mnli", device = device)
    
    parser = argparse.ArgumentParser(description='Filter norms dataset so that the moral_action/immoral_action entails/contradicts the rot_action.')
    parser.add_argument('-s','--subset_size', type=int, default=100, help='Size of the subset to process, -1 for full dataset')
    parser.add_argument('-e','--entailment_threshold', type=float, default=0.75, help='Minimum score for entailment')
    parser.add_argument('-c','--contradiction_threshold', type=float, default=0.85, help='Minimum score for contradiction')
    parser.add_argument('-b', '--batch_size', type=int, default=2000, help='Batch size for processing')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset')
    
    args = parser.parse_args()
    
    subset_size = args.subset_size
    entailment_threshold = args.entailment_threshold
    contradiction_threshold = args.contradiction_threshold
    batch_size = args.batch_size
    shuffle = args.shuffle
    
    edit_norms_subset = load_edit_norms(subset_size, shuffle)

    result = edit_norms_subset.filter(lambda batch: filter_moral_batch(batch, classifier, entailment_threshold, batch_size), batched=True, batch_size=batch_size)
    result = edit_norms_subset.filter(lambda batch: filter_immoral_batch(batch, classifier, contradiction_threshold, batch_size), batched=True, batch_size=batch_size)
    
    # Save it only if the full dataset was selected
    print(f"Number of neutral items: {len(result)}")
    result.to_json(f"{datasets_path}/norms/edit_norms_datasets/E{entailment_threshold}_C{contradiction_threshold}_S{subset_size}.json")