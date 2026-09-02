import re
import os

def parse_latex_table(latex_content):
    """
    Parses a specific LaTeX table format into a dictionary.
    Structure: { 'Dataset Name': [ {'method': 'Name', 'values': ['val1', 'val2'...]} ] }
    """
    data = {}
    current_dataset = None
    
    # Split into lines and filter empty ones
    lines = [line.strip() for line in latex_content.strip().split('\n') if line.strip()]
    
    for line in lines:
        # Skip standard latex commands that aren't data rows
        if line.startswith(r'\begin') or line.startswith(r'\end') or line.startswith(r'\textbf') or line.startswith(r'Evaluation'):
             continue
        if line.startswith(r'\hline'):
            continue
            
        # Remove trailing \\ and split by &
        clean_line = line.replace(r'\\', '')
        parts = [p.strip() for p in clean_line.split('&')]
        
        # Check if this is a Dataset header row (e.g., "Dataset Cancer")
        if len(parts) > 0 and parts[0].startswith('Dataset'):
            current_dataset = parts[0]
            if current_dataset not in data:
                 data[current_dataset] = []
            continue
            
        # Check if this is a data row (needs a valid dataset context and content)
        if current_dataset and len(parts) > 1:
            method_name = parts[0]
            # Store the raw latex strings for the values to preserve formatting (bolding, math mode)
            values = parts[1:] 
            data[current_dataset].append({
                'method': method_name,
                'values': values
            })
            
    return data

def get_float_value(latex_str):
    """Extracts the mean value from a string like '$2.00 {\scriptstyle \pm 0.45}$'"""
    # 1. Remove existing bolding and math markers to find the raw number
    clean = latex_str.replace('$', '').replace(r'\mathbf{', '').replace('}', '')
    
    # 2. Handle missing data entries (often represented as '-')
    if '-' in clean and len(clean) < 5: 
        return None
        
    # 3. Find the first floating point number in the string
    # This regex looks for digits, optionally followed by a dot and more digits
    match = re.search(r"(\d+\.\d+|\d+)", clean)
    if match:
        return float(match.group(1))
    return None

def format_bold(latex_str, should_be_bold):
    """
    Adds \mathbf{} around the main number if should_be_bold is True.
    Removes \mathbf{} if should_be_bold is False.
    """
    # 1. Strip outer dollar signs to work on the content
    inner = latex_str.strip('$')
    
    # 2. CLEANUP: Remove any existing \mathbf{} tags to start fresh
    # This replaces \mathbf{1.23} with just 1.23
    inner = re.sub(r'\\mathbf\{([\d\.]+)\}', r'\1', inner)
    
    # 3. If we don't want bold, just return the cleaned string wrapped in $
    if not should_be_bold:
        return f"${inner}$"
    
    # 4. APPLY BOLD: Wrap the first number found in \mathbf{}
    # The regex matches the start of the string (^) followed by digits/dots
    # It avoids bolding the \pm part.
    inner = re.sub(r'^([\d\.]+)', r'\\mathbf{\1}', inner.strip())
    
    return f"${inner}$"

def compare_and_format_rows(row1_vals, row2_vals):
    """
    Compares two lists of LaTeX value strings and bolds the best one.
    """
    new_r1 = []
    new_r2 = []
    
    # =========================================================
    # CONFIGURATION: Which columns are "Lower is Better"?
    # Indices correspond to: 
    # 0: SHD, 1: SHD_double, 2: SID_min, 3: SID_max (Errors -> Small is good)
    # 4: Precision, 5: Recall, 6: F1 (Scores -> Large is good)
    # =========================================================
    lower_is_better_indices = [0, 1, 2, 3]
    
    # Loop through each column
    for i in range(len(row1_vals)):
        val1_str = row1_vals[i]
        # Handle case where row2 might be shorter
        val2_str = row2_vals[i] if i < len(row2_vals) else "-"
        
        v1 = get_float_value(val1_str)
        v2 = get_float_value(val2_str)
        
        bold_v1 = False
        bold_v2 = False
        
        if v1 is not None and v2 is not None:
            if i in lower_is_better_indices:
                # -------------------------------
                # LOGIC FOR SHD/SID (Small is Good)
                # -------------------------------
                if v1 < v2: 
                    bold_v1 = True
                elif v2 < v1: 
                    bold_v2 = True
                else: 
                    # Tie
                    bold_v1 = True
                    bold_v2 = True
            else:
                # -------------------------------
                # LOGIC FOR PREC/REC/F1 (Large is Good)
                # -------------------------------
                if v1 > v2: 
                    bold_v1 = True
                elif v2 > v1: 
                    bold_v2 = True
                else:
                    # Tie
                    bold_v1 = True
                    bold_v2 = True
        
        # Handle edge cases where one value is missing (treat existing value as winner)
        elif v1 is not None and v2 is None:
            bold_v1 = True
        elif v2 is not None and v1 is None:
            bold_v2 = True

        # Apply formatting
        new_r1.append(format_bold(val1_str, bold_v1))
        new_r2.append(format_bold(val2_str, bold_v2))
        
    return new_r1, new_r2

def combine_tables_from_files(file_path1, file_path2, config1_name, config2_name, output_file=None):
    
    # Read files
    try:
        with open(file_path1, 'r', encoding='utf-8') as f:
            content1 = f.read()
        with open(file_path2, 'r', encoding='utf-8') as f:
            content2 = f.read()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    t1_data = parse_latex_table(content1)
    t2_data = parse_latex_table(content2)
    
    combined_latex = []
    
    # Table Header
    header = (
        "\\begin{tabular}{l|ccccccc}\n"
        "\\textbf{Evaluation Results} & SHD & SHD\\textsubscript{double} & "
        "SID\\textsubscript{min} & SID\\textsubscript{max} & Precision & Recall & F\\textsubscript{1} \\\\\n"
        "\\hline \\hline"
    )
    combined_latex.append(header)
    
    # Process Datasets
    for dataset, rows1 in t1_data.items():
        combined_latex.append(f"{dataset} & & & & & & & \\\\")
        combined_latex.append("\\hline")
        
        rows2 = t2_data.get(dataset, [])
        rows2_map = {r['method']: r['values'] for r in rows2}
        
        num_methods = len(rows1)
        
        for i, row1 in enumerate(rows1):
            method_base = row1['method']
            vals1_orig = row1['values']
            
            # Identify Config 1 and Config 2 values
            if method_base in rows2_map:
                vals2_orig = rows2_map[method_base]
                
                # COMPARE AND BOLD
                vals1_new, vals2_new = compare_and_format_rows(vals1_orig, vals2_orig)
                
                # Config 1 Row
                name_c1 = f"{method_base}{config1_name}"
                line_c1 = " & ".join(vals1_new)
                combined_latex.append(f"{name_c1} & {line_c1} \\\\")
                
                # Config 2 Row
                name_c2 = f"{method_base}{config2_name}"
                line_c2 = " & ".join(vals2_new)
                combined_latex.append(f"{name_c2} & {line_c2} \\\\")
                
            else:
                # If only present in file 1, print it as is (no comparison possible)
                name_c1 = f"{method_base}{config1_name}"
                line_c1 = " & ".join(vals1_orig)
                combined_latex.append(f"{name_c1} & {line_c1} \\\\")
                
                # Placeholder for missing file 2 data
                name_c2 = f"{method_base}{config2_name}"
                combined_latex.append(f"{name_c2} & - & - & - & - & - & - & - \\\\")

            # Add hline ONLY if this is NOT the last method for this dataset
            if i < num_methods - 1:
                combined_latex.append("\\hline")

        combined_latex.append("\\hline \\hline")
        
    # Cleanup last double hline
    if combined_latex[-1] == "\\hline \\hline":
        combined_latex.pop()
    
    combined_latex.append("\\hline \\hline")
    combined_latex.append("\\end{tabular}")
    
    final_output = "\n".join(combined_latex)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_output)
        print(f"Successfully wrote combined table to: {output_file}")
        
    return final_output

# ==========================================
# Usage Configuration
# ==========================================

path_to_file_1 = "results/main_evaluation/_result_tables/tagging_only_datasets.txt"
path_to_file_2 = "results/noise_evaluation/_result_tables/tagging_only_datasets.txt"
output_path    = "results/results_combined.tex"

name_for_file_1 = ""
name_for_file_2 = " (N)"

if __name__ == "__main__":
    if os.path.exists(path_to_file_1) and os.path.exists(path_to_file_2):
        combine_tables_from_files(
            path_to_file_1, 
            path_to_file_2, 
            name_for_file_1, 
            name_for_file_2, 
            output_path
        )
    else:
        print("Input files not found. Please check paths in 'Usage Configuration'.")