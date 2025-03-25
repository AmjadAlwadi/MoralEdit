import statistics
import json
import os
import re

from collections import defaultdict

precision = 2



def read_json_from_file(file):
    # Read the JSON file
    with open(file, 'r') as file:
        data = json.load(file)
        return data



def generate_latex_table(headers, rows):
    headers = [str(header) for header in headers]
    rows = [[str(value) for value in row] for row in rows]  # Fixed this

    # Start the table
    latex_table = "\\begin{tabular}{" + "|".join(["c"] * len(headers)) + "}\n"
    latex_table += "\\hline\n"
    
    # Add headers
    latex_table += " & ".join(headers) + " \\\\\n"
    latex_table += "\\hline\n"
    
    # Add rows
    for row in rows:
        latex_table += " & ".join(row) + " \\\\\n"
        latex_table += "\\hline\n"
    
    # End the table
    latex_table += "\\end{tabular}"
    
    return latex_table





def find_easy_edit_metric_files(root_dir):
    metric_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file == "post_edit_easy_edit_metrics.json":
                metric_files.append(os.path.join(dirpath, file))
    return metric_files





def find_sentiment_labels_metric_files(root_dir):
    metric_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file == "edit_effect_sentiment_labels_metric.json":
                metric_files.append(os.path.join(dirpath, file))
    return metric_files



def find_sentiment_scores_metric_files(root_dir):
    metric_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file == "edit_effect_sentiment_scores_metric.json":
                metric_files.append(os.path.join(dirpath, file))
    return metric_files





def find_kl_div_metric_files(root_dir):
    metric_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file == "edit_effect_kl_div_metric.json":
                metric_files.append(os.path.join(dirpath, file))
    return metric_files





def find_perplexity_metric_files(root_dir):
    metric_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file == "edit_effect_perplexity_metric.json":
                metric_files.append(os.path.join(dirpath, file))
    return metric_files





def get_metadata(file_path):
    
    # Get the directory of the file
    directory = os.path.dirname(file_path)
    json_file = os.path.join(directory, "metadata.json")
    
    if os.path.exists(json_file):
        return read_json_from_file(json_file)
    else:
        return None






def change_list_format(metric_files):
    
    new_metric_dict = {}
    
    for metric_file in metric_files:
        
        metric_file_config = metric_file[metric_file.find("outputs") + len("outputs\\"):].split("\\")
        editing_method = metric_file_config[0]
        model_name = metric_file_config[1]
        decoding_strategy = metric_file_config[2]
        number_of_sequential_edits = int(metric_file_config[3].split("_")[0])
        time_stamp = metric_file_config[4]
        metric_file_name = metric_file_config[5]
        
        metadata = get_metadata(metric_file)
        
        if editing_method == "IKE":
            if metadata["ike_demos_number"] == 0:
                editing_method = "PROMPTING"
            elif "ike_selection_mechanism" in metadata.keys() and metadata["ike_selection_mechanism"] == "similarity":
                editing_method = "IKE_S"
            else:
                editing_method = "IKE_R"
        
        if metadata["norms_dataset_number"] != 0:
            metadata["norms_dataset_number"] = 1
        
        new_metric_dict.update({metric_file : {"editing_method" : editing_method, "model_name" : model_name, "decoding_strategy" : decoding_strategy, "number_of_sequential_edits" : number_of_sequential_edits, "norms_dataset_number": metadata["norms_dataset_number"],  "time_stamp" : time_stamp, "dataset_number": metadata["norms_dataset_number"], "metric_file_name" : metric_file_name}})

    return new_metric_dict







def extract_kl_div_values(metric_path):
    
    metric_file = read_json_from_file(metric_path)
        
    kl_div_locality_neighborhood = metric_file["kl_div_locality_neighborhood"]
    kl_div_locality_distracting = metric_file["kl_div_locality_distracting"]
    

    edits_locality_neighborhood = [edit_item for edit_item in kl_div_locality_neighborhood]    
    edits_locality_distracting = [edit_item for edit_item in kl_div_locality_distracting] 
    
    
    edit_averages_locality_neighborhood_first_token = [] 
    edit_averages_locality_neighborhood_differing_token = [] 
    
    for edit in edits_locality_neighborhood:
        current_edit_average_first_token = statistics.mean([seq["kl_div_first_token"] for seq in edit])
        current_edit_average_differing_token = statistics.mean([seq["kl_div_differing_token"] for seq in edit])
        edit_averages_locality_neighborhood_first_token.append(current_edit_average_first_token)
        edit_averages_locality_neighborhood_differing_token.append(current_edit_average_differing_token)
        

    all_edits_average_locality_neighborhood_first_token = statistics.mean(edit_averages_locality_neighborhood_first_token)
    all_edits_average_locality_neighborhood_differing_token = statistics.mean(edit_averages_locality_neighborhood_differing_token)
    
    
    edit_averages_locality_distracting_first_token = [] 
    edit_averages_locality_distracting_differing_token = [] 
    
    for edit in edits_locality_distracting:
        current_edit_average_first_token = statistics.mean([seq["kl_div_first_token"] for seq in edit])
        current_edit_average_differing_token = statistics.mean([seq["kl_div_differing_token"] for seq in edit])
        edit_averages_locality_distracting_first_token.append(current_edit_average_first_token)
        edit_averages_locality_distracting_differing_token.append(current_edit_average_differing_token)
        

    all_edits_average_locality_distracting_first_token = statistics.mean(edit_averages_locality_distracting_first_token)
    all_edits_average_locality_distracting_differing_token = statistics.mean(edit_averages_locality_distracting_differing_token)
    
    
    kl_div_dict = {
        "kl_div_neighborhood_first_token_average" : all_edits_average_locality_neighborhood_first_token,
        "kl_div_neighborhood_differing_token_average" : all_edits_average_locality_neighborhood_differing_token,
        "kl_div_distracting_first_token_average" : all_edits_average_locality_distracting_first_token,
        "kl_div_distracting_differing_token_average" : all_edits_average_locality_distracting_differing_token, 
    }
    
    
        
    return kl_div_dict





def extract_perplexity_values(metric_path):
    metric_file = read_json_from_file(metric_path)
    abs_edits_locality_neighborhood = [item["absolute_locality_neighborhood_perplexity"] for item in metric_file]
    abs_edits_locality_distracting = [item["absolute_locality_distracting_perplexity"] for item in metric_file]
    rel_edits_locality_neighborhood = [item["relative_locality_neighborhood_perplexity"] for item in metric_file]
    rel_edits_locality_distracting = [item["relative_locality_distracting_perplexity"] for item in metric_file]

    def extract_last_list(s):
        pattern = r"\[([-]?\d*\.?\d+(?:\s*,\s*[-]?\d*\.?\d+)*)\]$"
        match = re.search(pattern, s)

        if match:
            result = [float(x) for x in match.group(1).split(',')]
            return result


    abs_edits_locality_neighborhood = [extract_last_list(item) for item in abs_edits_locality_neighborhood]
    abs_edits_locality_neighborhood_average = statistics.mean([statistics.mean(lst) for lst in abs_edits_locality_neighborhood])
    
    abs_edits_locality_distracting = [extract_last_list(item) for item in abs_edits_locality_distracting]
    abs_edits_locality_distracting_average = statistics.mean([statistics.mean(lst) for lst in abs_edits_locality_distracting])
    
    rel_edits_locality_neighborhood = [extract_last_list(item) for item in rel_edits_locality_neighborhood]
    rel_edits_locality_neighborhood_average = statistics.mean([statistics.mean(lst) for lst in rel_edits_locality_neighborhood])
    
    rel_edits_locality_distracting = [extract_last_list(item) for item in rel_edits_locality_distracting]
    rel_edits_locality_distracting_average = statistics.mean([statistics.mean(lst) for lst in rel_edits_locality_distracting])
    
    
    perplexity_dict = {
        "absolute_perplexity_neighborhood_average" : abs_edits_locality_neighborhood_average,
        "absolute_perplexity_distracting_average" : abs_edits_locality_distracting_average,
        "relative_perplexity_neighborhood_average" : rel_edits_locality_neighborhood_average,
        "relative_perplexity_distracting_average" : rel_edits_locality_distracting_average,
    }
    
    return perplexity_dict






def extract_sentiment_labels_values(metric_path):
    metric_file = read_json_from_file(metric_path)
    
    keys = ["prompt","light_generality_1","light_generality_2","light_generality_3","strong_generality","portability_synonym","portability_one_hop","portability_two_hop","locality_neighborhood","locality_distracting"]
    
    edits = {}
    edits_averages = {}
    
    for key in keys:
        edits[key] = [item[key] for item in metric_file]


    def extract_last_list(s):
        pattern = r"\[([0-9\.\s,]+)\]$"
        match = re.search(pattern, s)

        if match:
            result = [float(x) for x in match.group(1).split(',')]
            return result
        
        

    for key in keys:
        edits[key] = [extract_last_list(item) for item in edits[key]]
        edits_averages[key] = statistics.mean([statistics.mean(lst) for lst in edits[key]])
    
    
    sentiment_dict = {}
    
    for key in keys:
        sentiment_dict[f"sentiment_label_{key}_average"] = edits_averages[key]
    
    
    return sentiment_dict






def extract_sentiment_scores_values(metric_path):
    metric_file = read_json_from_file(metric_path)
    
    keys = ["prompt","light_generality_1","light_generality_2","light_generality_3","strong_generality","portability_synonym","portability_one_hop","portability_two_hop","locality_neighborhood","locality_distracting"]
    
    edits = {}
    edits_averages = {}
    
    for key in keys:
        edits[key] = [item[key] for item in metric_file]


    
    def extract_last_list(s):
        end_idx = s.rfind(']')
        if end_idx == -1:
            return None
        
        start_idx = s.rfind('[', 0, end_idx)
        if start_idx == -1:
            return None
        
        list_str = s[start_idx + 1:end_idx]
        
        try:
            result = [float(x.strip()) for x in list_str.split(',')]
            return result
        except ValueError:
            return None
        
        

    for key in keys:
        edits[key] = [extract_last_list(item) for item in edits[key]]
        edits_averages[key] = statistics.mean([statistics.mean(lst) for lst in edits[key]])
    
    
    sentiment_dict = {}
    
    for key in keys:
        sentiment_dict[f"sentiment_score_{key}_average"] = edits_averages[key]
    
    
    return sentiment_dict





def reduce_rows_to_averages(rows):
    grouped = defaultdict(list)

    # Group rows by (model, editing method)
    for row in rows:
        key = (row[0], row[1])  # First two columns (model, editing method)
        values = list(map(float, row[2:])) 
        grouped[key].append(values)

    # Average all columns for each (model, editing method) group
    reduced_rows = []
    for key, value_lists in grouped.items():
        # Transpose the list of lists to group by column
        columns = list(zip(*value_lists))
        # Compute the average for each column
        averages = [round(statistics.mean(column), precision) for column in columns]
        reduced_rows.append(list(key) + averages)

    return reduced_rows




def filter_dict_by(dictionary, conditions):
    
    filtered_dict = {
        key: value for key, value in dictionary.items()
        if all(value.get(k) == v for k, v in conditions.items())
    }


    return filtered_dict






def filter_by(conditions, kl_div_files_dict, sentiment_labels_files_dict, sentiment_scores_files_dict, perplexity_files_dict):

    kl_div_files_dict = filter_dict_by(kl_div_files_dict, conditions)
    sentiment_labels_files_dict = filter_dict_by(sentiment_labels_files_dict, conditions)
    sentiment_scores_files_dict = filter_dict_by(sentiment_scores_files_dict, conditions)
    perplexity_files_dict = filter_dict_by(perplexity_files_dict, conditions)

    return kl_div_files_dict, sentiment_labels_files_dict, sentiment_scores_files_dict, perplexity_files_dict




def change_underscore(s):
    return "-".join(s.split("_"))
