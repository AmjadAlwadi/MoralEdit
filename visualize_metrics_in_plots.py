import os
import statistics
import matplotlib.pyplot as plt


from collections import defaultdict
from visualization_utils import *

# import seaborn as sns
# sns.set_style("whitegrid")





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








# Caclulates the harmonic mean
def calculate_score(args):
    return statistics.mean(args)
    # return statistics.harmonic_mean(args)
        





def plot_sentiment_labels(rows, name = "plot"):
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
    plt.figure(figsize=(10, 6), dpi=300)  # High-resolution for publication

    # Plot each method
    for method, values in data_by_method.items():
        plt.plot(values['number_of_seq_edits'], values['scores'], 
                 marker='o',  # Points at each data point
                 linewidth=2,  # Thicker lines for visibility
                 markersize=6,  # Larger markers
                 label=method)

    # Customize the plot
    plt.xlabel('Number of Sequential Edits', fontsize=14)  # Larger x-axis label
    plt.ylabel('Mean Sentiment Score', fontsize=14)  # Larger y-axis label
    plt.title(f'Mean Sentiment Score as a Function of Number of Sequential Edits\nfor {title}', 
              fontsize=16, pad=15)  # Larger title, multi-line
    plt.xticks(fontsize=12)  # Larger tick labels
    plt.yticks(fontsize=12)  # Larger tick labels
    plt.legend(fontsize=14, loc='upper left', bbox_to_anchor=(1.05, 1), 
               frameon=True, edgecolor='black')  # Readable legend outside plot
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save for publication
    plt.savefig(f'sequential_edits/sentiment_score/{name}.pdf', bbox_inches='tight', dpi=300)  # PDF for LaTeX

    # Show the plot (optional for development)
    # plt.show()









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
    plt.title('Mean Sentiment Magnitude vs Sequential Edits Number by Editing Method')
    plt.legend()  # adds the legend to distinguish methods
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    # plt.show()
    
    
    
    
    
    
    
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
    plt.title('Mean Sentiment Magnitude for Locality vs Sequential Edits Number by Editing Method')
    plt.legend()  # adds the legend to distinguish methods
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    # plt.show()








def plot_perplexity_scores(rows, name="plot", log=False):
    from collections import defaultdict
    import matplotlib.pyplot as plt

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
    plt.figure(figsize=(10, 6), dpi=300)  # High-resolution for publication

    # Plot each method
    for method, values in data_by_method.items():
        plt.plot(values['number_of_seq_edits'], values['scores'], 
                 marker='o',  # Points at each data point
                 linewidth=2,  # Thicker lines for visibility
                 markersize=6,  # Larger markers
                 label=method)

    # Set logarithmic scale for y-axis
    if log:
        plt.yscale('log')

    # Customize the plot
    plt.xlabel('Number of Sequential Edits', fontsize=14)  # Larger x-axis label
    plt.ylabel('Mean Absolute Perplexity Score (Log Scale)', fontsize=14)  # Updated y-axis label
    plt.title(f'Mean Absolute Perplexity Score as a Function of Number of Sequential Edits\nfor {name}', 
              fontsize=16, pad=15)  # Larger title, multi-line
    plt.xticks(fontsize=12)  # Larger tick labels
    plt.yticks(fontsize=12)  # Larger tick labels
    plt.legend(fontsize=14, loc='upper left', bbox_to_anchor=(1.05, 1), 
               frameon=True, edgecolor='black')  # Readable legend outside plot
    plt.grid(True, linestyle='--', alpha=0.7)  # Grid for major ticks only

    # Save for publication
    plt.savefig(f'sequential_edits/perplexity_score/{name}.pdf', bbox_inches='tight', dpi=300)  # PDF for LaTeX

    # Show the plot (optional for development)
    # plt.show()
    
    
    
    
    
def plot_kl_div_scores(rows, name="plot"):
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
    plt.figure(figsize=(10, 6), dpi=300)  # High-resolution for publication

    # Plot each method
    for method, values in data_by_method.items():
        plt.plot(values['number_of_seq_edits'], values['scores'], 
                 marker='o',  # Points at each data point
                 linewidth=2,  # Thicker lines for visibility
                 markersize=6,  # Larger markers
                 label=method)

    # Customize the plot
    plt.xlabel('Number of Sequential Edits', fontsize=12)  # Larger x-axis label
    plt.ylabel('Mean KL Divergence Score', fontsize=12)  # Larger y-axis label
    plt.title(f'Mean KL Divergence Score as a Function of Number of Sequential Edits\nfor {title}', 
              fontsize=16, pad=15)  # Larger title, multi-line
    plt.xticks(fontsize=12)  # Larger tick labels
    plt.yticks(fontsize=12)  # Larger tick labels
    plt.legend(fontsize=14, loc='upper left', bbox_to_anchor=(1.05, 1), 
               frameon=True, edgecolor='black')  # Readable legend outside plot
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save for publication
    plt.savefig(f'sequential_edits/kl_div_score/{name}.pdf', bbox_inches='tight', dpi=300)  # PDF for LaTeX

    # Show the plot (optional for development)
    # plt.show()







def get_rows():
    
    kl_div_files_dict = change_list_format(find_kl_div_metric_files(os.getcwd()))
    sentiment_labels_files_dict = change_list_format(find_sentiment_labels_metric_files(os.getcwd()))
    sentiment_scores_files_dict = change_list_format(find_sentiment_scores_metric_files(os.getcwd()))
    perplexity_files_dict = change_list_format(find_perplexity_metric_files(os.getcwd()))
    
    
    kl_div_rows = []
    perplexity_rows = []
    sentiment_labels_rows = []
    sentiment_scores_rows = []
    

    
    # Coherent dataset
    filter_conditions = {
        "model_name" : "gpt2-xl",
        "decoding_strategy" : "beam-search multinomial sampling",
        "norms_dataset_number": 1
    }
    
    # # Random dataset
    # filter_conditions = {
    #     "model_name" : "gpt2-xl",
    #     "decoding_strategy" : "beam-search multinomial sampling",
    #     "norms_dataset_number": 0
    # }
    
    
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

    # Methods, that don't support sequential editing
    methods_to_remove = ["LORA", "MELO", "IKE_S"]
    
    methods_to_remove += ["GRACE", "ICE", "PROMPT", "IKE_R", "MEND", "R-ROME", "WISE"]
    methods_to_remove += ["FT-L", "FT-M", "ROME"]
    # methods_to_remove += ["PROMPT"]
    

    # # Locate-Then-Edit
    # for method in ["FT-L", "FT-M", "ROME", "R-ROME"]:
    #     methods_to_remove.remove(method)
    
    # Memory-based
    for method in ["IKE_R", "ICE", "PROMPT", "GRACE", "WISE"]:
        methods_to_remove.remove(method)
    
    # # Meta-learning
    # for method in ["MEND"]:
    #     methods_to_remove.remove(method)
    
    
    
    sentiment_labels_rows = remove_editing_methods(sentiment_labels_rows, methods_to_remove)
    sentiment_scores_rows = remove_editing_methods(sentiment_scores_rows, methods_to_remove)
    kl_div_rows = remove_editing_methods(kl_div_rows, methods_to_remove)
    perplexity_rows = remove_editing_methods(perplexity_rows, methods_to_remove)
    
    
    return sentiment_labels_rows, sentiment_scores_rows, kl_div_rows, perplexity_rows




if __name__ == "__main__":
    
    global title
    
    # title = "Locate-Then-Edit Methods"
    title = "Memory-based Methods"
    # title = "Meta-learning Methods"
    # title = "All Methods"
    
    # Rows extraction
    sentiment_labels_rows, sentiment_scores_rows, kl_div_rows, perplexity_rows = get_rows()
    
    
    # Plotting
    # plot_sentiment_scores(sentiment_scores_rows)
    # plot_sentiment_loc_scores(sentiment_scores_rows)
    
    plot_name = "memory-based"
    
    plot_sentiment_labels(sentiment_labels_rows, f"{plot_name}_plot")
    plot_perplexity_scores(perplexity_rows, f"{plot_name}_plot", False)
    plot_kl_div_scores(kl_div_rows, f"{plot_name}_plot")
    

    
