import argparse
import json
import os
import numpy as np
import statistics
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# TODO:
# sort according to final score





kl_div_files_dict = None
sentiment_files_dict = None
perplexity_files_dict = None
headers = None

keep_editing_method=True
keep_model_name=True
keep_decoding_method = True
keep_number_of_sequential_edits = True

precision = 2


# Generate a latex table and a plot

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
        
        new_metric_dict.update({metric_file : {"editing_method" : editing_method, "model_name" : model_name, "decoding_strategy" : decoding_strategy, "number_of_sequential_edits" : number_of_sequential_edits, "time_stamp" : time_stamp, "metric_file_name" : metric_file_name}})

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









def format_kl_div_as_row(metric_configuration, metric_value):
    '''
    returns the seperate values and an average score
    
    '''
    
    score = calculate_score([metric_value["kl_div_neighborhood_first_token_average"],
                            metric_value["kl_div_neighborhood_differing_token_average"],
                            metric_value["kl_div_distracting_first_token_average"],
                            metric_value["kl_div_distracting_differing_token_average"]])
    
    return [
        metric_configuration["editing_method"],
        metric_configuration["number_of_sequential_edits"],
        score
    ]





def format_perplexity_as_row(metric_configuration, metric_value):
    '''
    returns the seperate values and an average score
    
    '''
    
    score = calculate_score([metric_value["absolute_perplexity_neighborhood_average"],
                            metric_value["absolute_perplexity_distracting_average"]])
    
    
    return [
        metric_configuration["editing_method"],
        metric_configuration["number_of_sequential_edits"],
        score
    ]






def format_sentiment_labels_as_row(metric_configuration, metric_value):
    
    '''
    returns the average score
    
    '''
    light_generality = statistics.mean([metric_value["sentiment_label_light_generality_1_average"], metric_value["sentiment_label_light_generality_2_average"], metric_value["sentiment_label_light_generality_3_average"]])
    
    score = calculate_score([metric_value["sentiment_label_prompt_average"],
                            light_generality, metric_value["sentiment_label_strong_generality_average"],
                            metric_value["sentiment_label_portability_synonym_average"],
                            metric_value["sentiment_label_portability_one_hop_average"],
                            metric_value["sentiment_label_portability_two_hop_average"],
                            1 - metric_value["sentiment_label_locality_neighborhood_average"],
                            1 - metric_value["sentiment_label_locality_distracting_average"]])
    
    return [
        metric_configuration["editing_method"],
        metric_configuration["number_of_sequential_edits"],
        score 
    ]






def format_sentiment_scores_as_row(metric_configuration, metric_value):
    
    '''
    returns the average score of ↑ metrics and the average of the ↓ metrics (locality)
    
    '''
    
    light_generality = statistics.mean([metric_value["sentiment_score_light_generality_1_average"], metric_value["sentiment_score_light_generality_2_average"], metric_value["sentiment_score_light_generality_3_average"]])
    
    score = statistics.mean([metric_value["sentiment_score_prompt_average"],
                            light_generality,
                            metric_value["sentiment_score_strong_generality_average"],
                            metric_value["sentiment_score_portability_synonym_average"],
                            metric_value["sentiment_score_portability_one_hop_average"],
                            metric_value["sentiment_score_portability_two_hop_average"]])
    
    loc_score = statistics.mean([metric_value["sentiment_score_locality_neighborhood_average"],
                                metric_value["sentiment_score_locality_distracting_average"]])
    
    
    return [
        metric_configuration["editing_method"],
        metric_configuration["number_of_sequential_edits"],
        score,
        loc_score
    ]






def change_underscore(s):
    return "-".join(s.split("_"))






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






# Caclulates the harmonic mean
def calculate_score(args):
    return statistics.mean(args)
    # return statistics.harmonic_mean(args)
        






def get_rows():
    
    kl_div_files_dict = change_list_format(find_kl_div_metric_files(os.getcwd()))
    sentiment_labels_files_dict = change_list_format(find_sentiment_labels_metric_files(os.getcwd()))
    sentiment_scores_files_dict = change_list_format(find_sentiment_scores_metric_files(os.getcwd()))
    perplexity_files_dict = change_list_format(find_perplexity_metric_files(os.getcwd()))
    
    
    kl_div_rows = []
    perplexity_rows = []
    sentiment_labels_rows = []
    sentiment_scores_rows = []
    
    
    filter_conditions = {
        # "editing_method" : "MEND",
        "model_name" : "gpt2-xl",
        "decoding_strategy" : "beam-search multinomial sampling",
        # "number_of_sequential_edits" : seq_edit_number
    }
    
    
    if len(filter_conditions) > 0:
        kl_div_files_dict, sentiment_labels_files_dict, sentiment_scores_files_dict, perplexity_files_dict = filter_by(filter_conditions, kl_div_files_dict, sentiment_labels_files_dict, sentiment_scores_files_dict, perplexity_files_dict)
    
    

    for metric_path, metric_configuration in sentiment_labels_files_dict.items():
        result = extract_sentiment_labels_values(metric_path)
        sentiment_labels_rows.append(format_sentiment_labels_as_row(metric_configuration, result))
        
        
    for metric_path, metric_configuration in sentiment_scores_files_dict.items():
        result = extract_sentiment_scores_values(metric_path)
        sentiment_scores_rows.append(format_sentiment_scores_as_row(metric_configuration, result))
    
    
    for metric_path, metric_configuration in kl_div_files_dict.items():
        result = extract_kl_div_values(metric_path)
        kl_div_rows.append(format_kl_div_as_row(metric_configuration, result))
    
    
    for metric_path, metric_configuration in perplexity_files_dict.items():
        result = extract_perplexity_values(metric_path)
        perplexity_rows.append(format_perplexity_as_row(metric_configuration, result))
    
    
    sentiment_labels_rows = reduce_rows_to_averages(sentiment_labels_rows)
    sentiment_scores_rows = reduce_rows_to_averages(sentiment_scores_rows)
    kl_div_rows = reduce_rows_to_averages(kl_div_rows)
    perplexity_rows = reduce_rows_to_averages(perplexity_rows)


    return sentiment_labels_rows, sentiment_scores_rows, kl_div_rows, perplexity_rows






def plot_sentiment_labels(rows):

    # Organize data by editing method
    data_by_method = defaultdict(lambda: {'number_of_seq_edits': [], 'scores': []})
    for method, number_of_seq_edits, score in rows:
        data_by_method[method]['number_of_seq_edits'].append(number_of_seq_edits)
        data_by_method[method]['scores'].append(score)
            
    # Sort the data for each method by the number of sequential edits
    for method in data_by_method:
        paired = list(zip(data_by_method[method]['number_of_seq_edits'], data_by_method[method]['scores']))
        paired_sorted = sorted(paired, key=lambda x: x[0])
        data_by_method[method]['number_of_seq_edits'], data_by_method[method]['scores'] = zip(*paired_sorted)
    
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each method
    for method, values in data_by_method.items():
        plt.plot(values['number_of_seq_edits'], values['scores'], 
                marker='o',  # adds points at each data point
                label=method)

    # Customize the plot
    plt.xlabel('Number of Sequential Edits')
    plt.ylabel('Score')
    plt.title('Mean Sentiment Score vs Sequential Edits Number by Editing Method')
    plt.legend()  # adds the legend to distinguish methods
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()







def plot_sentiment_scores(rows):

    # Organize data by editing method
    data_by_method = defaultdict(lambda: {'number_of_seq_edits': [], 'scores': []})
    for method, number_of_seq_edits, score, loc_score in rows:
        data_by_method[method]['number_of_seq_edits'].append(number_of_seq_edits)
        data_by_method[method]['scores'].append(score)
    
    # Sort the data for each method by the number of sequential edits
    for method in data_by_method:
        paired = list(zip(data_by_method[method]['number_of_seq_edits'], data_by_method[method]['scores']))
        paired_sorted = sorted(paired, key=lambda x: x[0])
        data_by_method[method]['number_of_seq_edits'], data_by_method[method]['scores'] = zip(*paired_sorted)
    
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each method
    for method, values in data_by_method.items():
        plt.plot(values['number_of_seq_edits'], values['scores'], 
                marker='o',  # adds points at each data point
                label=method)

    # Customize the plot
    plt.xlabel('Number of Sequential Edits')
    plt.ylabel('Score')
    plt.title('Mean Sentiment Score Difference vs Sequential Edits Number by Editing Method')
    plt.legend()  # adds the legend to distinguish methods
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()
    
    
    
    
    
    
    
def plot_sentiment_loc_scores(rows):

    # Organize data by editing method
    data_by_method = defaultdict(lambda: {'number_of_seq_edits': [], 'scores': []})
    for method, number_of_seq_edits, score, loc_score in rows:
        data_by_method[method]['number_of_seq_edits'].append(number_of_seq_edits)
        data_by_method[method]['scores'].append(loc_score)
    
    # Sort the data for each method by the number of sequential edits
    for method in data_by_method:
        paired = list(zip(data_by_method[method]['number_of_seq_edits'], data_by_method[method]['scores']))
        paired_sorted = sorted(paired, key=lambda x: x[0])
        data_by_method[method]['number_of_seq_edits'], data_by_method[method]['scores'] = zip(*paired_sorted)
    
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each method
    for method, values in data_by_method.items():
        plt.plot(values['number_of_seq_edits'], values['scores'], 
                marker='o',  # adds points at each data point
                label=method)

    # Customize the plot
    plt.xlabel('Number of Sequential Edits')
    plt.ylabel('Score')
    plt.title('Mean Sentiment Score Difference for Locality vs Sequential Edits Number by Editing Method')
    plt.legend()  # adds the legend to distinguish methods
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()








def plot_perplexity_scores(rows):

    # Organize data by editing method
    data_by_method = defaultdict(lambda: {'number_of_seq_edits': [], 'scores': []})
    for method, number_of_seq_edits, score in rows:
        data_by_method[method]['number_of_seq_edits'].append(number_of_seq_edits)
        data_by_method[method]['scores'].append(score)
    
    # Sort the data for each method by the number of sequential edits
    for method in data_by_method:
        paired = list(zip(data_by_method[method]['number_of_seq_edits'], data_by_method[method]['scores']))
        paired_sorted = sorted(paired, key=lambda x: x[0])
        data_by_method[method]['number_of_seq_edits'], data_by_method[method]['scores'] = zip(*paired_sorted)
    
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each method
    for method, values in data_by_method.items():
        plt.plot(values['number_of_seq_edits'], values['scores'], 
                marker='o',  # adds points at each data point
                label=method)

    # Customize the plot
    plt.xlabel('Number of Sequential Edits')
    plt.ylabel('Score')
    plt.title('Mean Relative Perplexity Score vs Sequential Edits Number by Editing Method')
    plt.legend()  # adds the legend to distinguish methods
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()
    
    
    
    
    
    
def plot_kl_div_scores(rows):

    # Organize data by editing method
    data_by_method = defaultdict(lambda: {'number_of_seq_edits': [], 'scores': []})
    for method, number_of_seq_edits, score in rows:
        data_by_method[method]['number_of_seq_edits'].append(number_of_seq_edits)
        data_by_method[method]['scores'].append(score)
    
    # Sort the data for each method by the number of sequential edits
    for method in data_by_method:
        paired = list(zip(data_by_method[method]['number_of_seq_edits'], data_by_method[method]['scores']))
        paired_sorted = sorted(paired, key=lambda x: x[0])
        data_by_method[method]['number_of_seq_edits'], data_by_method[method]['scores'] = zip(*paired_sorted)
    
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each method
    for method, values in data_by_method.items():
        plt.plot(values['number_of_seq_edits'], values['scores'], 
                marker='o',  # adds points at each data point
                label=method)

    # Customize the plot
    plt.xlabel('Number of Sequential Edits')
    plt.ylabel('Score')
    plt.title('Mean KL Divergence Score vs Sequential Edits Number by Editing Method')
    plt.legend()  # adds the legend to distinguish methods
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()








if __name__ == "__main__":
    
    sentiment_labels_rows, sentiment_scores_rows, kl_div_rows, perplexity_rows = get_rows()
    # plot_sentiment_labels(sentiment_labels_rows)
    # plot_sentiment_scores(sentiment_scores_rows)
    # plot_sentiment_loc_scores(sentiment_scores_rows)
    # plot_perplexity_scores(perplexity_rows)
    # plot_kl_div_scores(kl_div_rows)
    

    
