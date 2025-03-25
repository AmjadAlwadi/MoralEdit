import os
import statistics
import matplotlib.pyplot as plt


from collections import defaultdict
from visualization_utils import *



# TODO:
# sort according to final score





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


    return sentiment_labels_rows, sentiment_scores_rows, kl_div_rows, perplexity_rows




if __name__ == "__main__":
    
    # Rows extraction
    sentiment_labels_rows, sentiment_scores_rows, kl_div_rows, perplexity_rows = get_rows()
    
    # Plotting
    plot_sentiment_labels(sentiment_labels_rows)
    # plot_sentiment_scores(sentiment_scores_rows)
    # plot_sentiment_loc_scores(sentiment_scores_rows)
    # plot_perplexity_scores(perplexity_rows)
    # plot_kl_div_scores(kl_div_rows)
    

    
