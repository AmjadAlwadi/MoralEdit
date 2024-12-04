import json
import argparse
import numpy as np
import os

# Generate a latex table and a plot

def read_json_from_file(file):
    # Read the JSON file
    with open(file, 'r') as file:
        data = json.load(file)
        return data



def generate_latex_table(headers, rows):
    
    headers = [str(header) for header in headers]
    rows = [[str(*value) for value in row] for row in rows]
    
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




def extract_leaf_keys(data, parent_key=''):
    keys_list = []
    values_list = []
    for k, v in data.items():
        k = k.replace('_', '-')
        full_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            sub_keys, sub_values = extract_leaf_keys(v, full_key)
            keys_list.extend(sub_keys)
            values_list.extend(sub_values)
        else:
            keys_list.append(full_key)
            values_list.append(v)
    return keys_list, values_list





def average_among_single_file(data, pre=False):
    
    values_list = []
    key = "post"
    
    if pre:
        key = "pre"
    
    
    for edit in data:
        _, values = extract_leaf_keys(edit[key])
        values_list.append(values)
    
    
    # Convert into np arrays
    values_list = [np.array(values) for values in values_list]
    values_list = np.array(values_list)
    
    average_values = np.mean(values_list, axis=0)

    return average_values.tolist()





def find_metric_files(root_dir):
    metric_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file == "metrics_summary.json":
                metric_files.append(os.path.join(dirpath, file))
    return metric_files




def find_custom_metric_files(root_dir):
    metric_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file == "custom_metric.json":
                metric_files.append(os.path.join(dirpath, file))
    return metric_files






def average_among_all_files(custom_metric=False, pre=False, *method):
    
    files = []
    values_for_each_file = []
    
    if custom_metric:
        files = find_custom_metric_files(os.getcwd())
    else:
        files = find_metric_files(os.path.join(os.getcwd(),"outputs",*method))
        
    files = [read_json_from_file(file) for file in files]
    
    if len(files) == 0:
        return None
    
    for file in files:
        if file:
            values_for_each_file.append(average_among_single_file(file,pre))

    values_for_each_file = np.array(values_for_each_file)
    average_values_among_all_files = np.mean(values_for_each_file, axis=0)
    
    return average_values_among_all_files


if __name__ == "__main__":
    
    
    # Extract headers
    metrics_data = read_json_from_file("outputs/ROME/gpt2-xl/greedy-decoding/02-12-2024__20-11/metrics_summary.json")
    custom_metrics_data = read_json_from_file("outputs/ROME/gpt2-xl/greedy-decoding/02-12-2024__20-11/metrics_summary.json")

    metrics_headers, _ = extract_leaf_keys(metrics_data[0]['pre'])
    custom_metrics_headers, _ = extract_leaf_keys(custom_metrics_data[0]['pre'])
    
    # Create for every method
    rome_average = average_among_all_files(False, False, "ROME")
    rome_average_gpt = average_among_all_files(False, False,"ROME","gpt2-xl")
    # ike_average = average_among_all_files(False, False,)
    # mend_average = average_among_all_files(False, False)
    # rome_average = average_among_all_files(False, False)
    

    # Generate the LaTeX table
    latex_table = generate_latex_table(metrics_headers, [rome_average])
    
    # Print the LaTeX table
    print(latex_table)