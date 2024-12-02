import json

# Generate a latex table and a plot

def read_json_from_file(file):
    # Read the JSON file
    with open(file, 'r') as file:
        data = json.load(file)
        return data



def generate_latex_table(headers, rows):
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






