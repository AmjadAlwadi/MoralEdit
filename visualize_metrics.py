import argparse
import json
import os
import numpy as np
import statistics
import re

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








def find_metric_files(root_dir):
    metric_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file == "metrics_summary.json":
                metric_files.append(os.path.join(dirpath, file))
    return metric_files




def find_sentiment_metric_files(root_dir):
    metric_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file == "edit_effect_sentiment_metric.json":
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





def extract_kl_values(metric_path):
    
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
        "neighborhood_first_token_average" : all_edits_average_locality_neighborhood_first_token,
        "neighborhood_differing_token_average" : all_edits_average_locality_neighborhood_differing_token,
        "distracting_first_token_average" : all_edits_average_locality_distracting_first_token,
        "distracting_differing_token_average" : all_edits_average_locality_distracting_differing_token, 
    }
    
    
        
    return kl_div_dict





def extract_perplexity_values(metric_path):
    metric_file = read_json_from_file(metric_path)
    edits_locality_neighborhood = [item["locality_neighborhood_perplexity"] for item in metric_file]
    edits_locality_distracting = [item["locality_distracting_perplexity"] for item in metric_file]

    def extract_last_list(s):
        pattern = r"\[([\d\.\s,]+)\]$"
        match = re.search(pattern, s)

        if match:
            result = [float(x) for x in match.group(1).split(',')]
            return result


    edits_locality_neighborhood = [extract_last_list(item) for item in edits_locality_neighborhood]
    edits_locality_neighborhood_average = statistics.mean([statistics.mean(lst) for lst in edits_locality_neighborhood])
    
    edits_locality_distracting = [extract_last_list(item) for item in edits_locality_distracting]
    edits_locality_distracting_average = statistics.mean([statistics.mean(lst) for lst in edits_locality_distracting])
    
    perplexity_dict = {
        "edits_locality_neighborhood_average" : edits_locality_neighborhood_average,
        "edits_locality_distracting_average" : edits_locality_distracting_average,
    }
    
    return perplexity_dict






def extract_sentiment_values(metric_path):
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
        sentiment_dict[f"edits_{key}_average"] = edits_averages[key]
    
    
    return sentiment_dict










# def average_among_all_files(custom_metric=False, pre=False, *method):
    
#     files = []
#     values_for_each_file = []
    
#     if custom_metric:
#         files = find_custom_metric_files(os.getcwd())
#     else:
#         files = find_metric_files(os.path.join(os.getcwd(),"outputs",*method))
        
#     files = [read_json_from_file(file) for file in files]
    
#     if len(files) == 0:
#         return None
    
#     for file in files:
#         if file:
#             values_for_each_file.append(average_among_single_file(file,pre))

#     values_for_each_file = np.array(values_for_each_file)
#     average_values_among_all_files = np.mean(values_for_each_file, axis=0)
    
#     return average_values_among_all_files






# def extract_leaf_keys(data, parent_key=''):
#     keys_list = []
#     values_list = []
#     for k, v in data.items():
#         k = k.replace('_', '-')
#         full_key = f"{parent_key}.{k}" if parent_key else k
#         if isinstance(v, dict):
#             sub_keys, sub_values = extract_leaf_keys(v, full_key)
#             keys_list.extend(sub_keys)
#             values_list.extend(sub_values)
#         else:
#             keys_list.append(full_key)
#             values_list.append(v)
#     return keys_list, values_list





# def average_among_single_file(data, pre=False):
    
#     values_list = []
#     key = "post"
    
#     if pre:
#         key = "pre"
    
    
#     for edit in data:
#         _, values = extract_leaf_keys(edit[key])
#         values_list.append(values)
    
    
#     # Convert into np arrays
#     values_list = [np.array(values) for values in values_list]
#     values_list = np.array(values_list)
    
#     average_values = np.mean(values_list, axis=0)

#     return average_values.tolist()



def format_as_row(value, result_key, result_value):
    return [change_underscore(result_key), value["editing_method"], value["model_name"], value["decoding_strategy"], str(value["number_of_sequential_edits"]), f"{result_value:.4f}"]




def change_underscore(s):
    return "-".join(s.split("_"))




if __name__ == "__main__":
    
    
    # Extract headers
    # metrics_data = read_json_from_file("outputs/ROME/gpt2-xl/greedy-decoding/02-12-2024__20-11/metrics_summary.json")
    # custom_metrics_data = read_json_from_file("outputs/ROME/gpt2-xl/greedy-decoding/02-12-2024__20-11/metrics_summary.json")

    # metrics_headers, _ = extract_leaf_keys(metrics_data[0]['pre'])
    # custom_metrics_headers, _ = extract_leaf_keys(custom_metrics_data[0]['pre'])
    
    # Create for every method
    # rome_average = average_among_all_files(False, False, "ROME")
    # rome_average_gpt = average_among_all_files(False, False,"ROME","gpt2-xl")
    # ike_average = average_among_all_files(False, False,)
    # mend_average = average_among_all_files(False, False)
    # rome_average = average_among_all_files(False, False)
    

    # for key, value in change_list_format(find_kl_div_metric_files(os.getcwd())).items():
    #     print(extract_kl_values(key))
    
    # for key, value in change_list_format(find_kl_div_metric_files(os.getcwd())).items():
    #     print(extract_kl_values(key))
    #     print("*"*5)
        
        
    # for key, value in change_list_format(find_perplexity_metric_files(os.getcwd())).items():
    #     extract_perplexity_values(key)
    #     print("*"*5)
        
        
    # for key, value in change_list_format(find_sentiment_metric_files(os.getcwd())).items():
    #     print(extract_sentiment_values(key))
    #     print("*"*5)
        
        

    kl_div_files_dict = change_list_format(find_kl_div_metric_files(os.getcwd()))
    sentiment_files_dict = change_list_format(find_sentiment_metric_files(os.getcwd()))
    perplexity_files_dict = change_list_format(find_perplexity_metric_files(os.getcwd()))
        
        
    headers = ["Metric", "Editing Method", "Model", "Decoding Strategy", "Sequential Edits", "value"]
    
    rows = []
    
    for key, value in kl_div_files_dict.items():
        result = extract_kl_values(key)
        
        for result_key, result_value in result.items():
            rows.append(format_as_row(value, result_key, result_value))
   
    
    
    for key, value in perplexity_files_dict.items():
        result = extract_perplexity_values(key)
        
        for result_key, result_value in result.items():
            rows.append(format_as_row(value, result_key, result_value))
        
        
    for key, value in sentiment_files_dict.items():
        result = extract_sentiment_values(key)
        
        for result_key, result_value in result.items():
            rows.append(format_as_row(value, result_key, result_value))
    
        

    # Generate the LaTeX table
    latex_table = generate_latex_table(headers, rows)
    
    # # Print the LaTeX table
    print(latex_table)