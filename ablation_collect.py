import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

custom_palette = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', 
                '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC', '#493267']


def format_k(n):
    if n < 1000:
        return str(int(round(n)))
    
    k_val = n / 1000
    # Round to 1 decimal place
    rounded = round(k_val, 1)
    
    # If it's a whole number (like 10.0), show as 10k
    # Except for 1.0k as per your specific requirement
    if rounded == 1.0:
        return "1.0k"
    if rounded == int(rounded):
        return f"{int(rounded)}k"
    
    return f"{rounded}k"

def table(ablation_type, seed):
    type_path = base_path + f"{ablation_type}/"
    save_path = base_save_path + f"{ablation_type}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for llm in llms:
        data = {}
        for i in range(1, 6):
            # load csv
            df = pd.read_csv(f"{type_path}{i}_{seed}/{llm}.csv")
            for dataset in datasets:
                correct = df.loc[df['dataset'] == dataset, 'tagging_correct'].sum()
                incorrect = df.loc[df['dataset'] == dataset, 'tagging_incorrect'].sum()
                undirected = df.loc[df['dataset'] == dataset, 'tagging_nothing'].sum()
                # accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else np.nan
                # samples = correct + incorrect
                if dataset not in data:
                    data[dataset] = []
                # data[dataset].append((accuracy, samples))
                data[dataset].append((correct, incorrect, undirected))

        def acc_string(value, sample):
            if np.isnan(value):
                return "-"
            value_str = f"{value:.2f}"
            if sample == 1:
                value_str += "*"
            elif sample < 5:
                value_str += "$^\circ$"
            return value_str
        
        def values_string(correct, incorrect, undirected):
            correct_str = str(correct)
            incorrect_str = str(incorrect)
            undirected_str = str(undirected)
            return f"{correct_str} / {incorrect_str} / {undirected_str}"

        # make table
        with open(f"{save_path}/{llm}.txt", "w") as f:
            f.write("\\begin{tabular}{l|ccccccccccc}\n")
            data_str = " & ".join([dataset_names[dataset][:2] for dataset in datasets])
            f.write(f" & {data_str} \\\\\n")
            f.write("\\hline\n")
            for i in range(1, 6):
                values = [data[dataset][i - 1] for dataset in datasets]
                
                # We assume values_string needs to be updated to use format_k
                # Here is the logic inside the join:
                formatted_results = []
                for correct, incorrect, undirected in values:
                    # Format each part of the tuple using the helper
                    c_str = format_k(correct)
                    i_str = format_k(incorrect)
                    u_str = format_k(undirected)
                    
                    # This calls your original values_string logic but with the new strings
                    # Adjust this line if your values_string function is defined differently
                    formatted_results.append(values_string(c_str, i_str, u_str))
                
                values_str = " & ".join(formatted_results)
                f.write(f" & {i*10}\% & {values_str} \\\\\n")
                f.write("\\end{tabular}\n")

    # and one big table that include all llms
    with open(f"{save_path}/all.txt", "w") as f:
        f.write("\\begin{tabular}{cl|ccccccccccc}\n")
        data_str = " & ".join([dataset_names[dataset][:2] for dataset in datasets])
        f.write(f" & & {data_str} \\\\\n")
        f.write("\\hline\n")
        for llm in llms:
            data = {}
            for i in range(1, 6):
                # load csv
                df = pd.read_csv(f"{type_path}{i}_{seed}/{llm}.csv")
                for dataset in datasets:
                    correct = df.loc[df['dataset'] == dataset, 'tagging_correct'].sum()
                    incorrect = df.loc[df['dataset'] == dataset, 'tagging_incorrect'].sum()
                    undirected = df.loc[df['dataset'] == dataset, 'tagging_nothing'].sum()
                    if dataset not in data:
                        data[dataset] = []
                    data[dataset].append((correct, incorrect, undirected))

            f.write("\\hline\n")
            llm_text = llm_to_text[llm]
            f.write(f"\\multirow{{5}}{{*}}{{\\rotatebox{{90}}{{{llm_text}}}}}")
            for i in range(1, 6):
                values = [data[dataset][i - 1] for dataset in datasets]
                # Apply format_val to each of the three variables before passing to values_string
                values_str = " & ".join([
                    values_string(format_k(c), format_k(inc), format_k(u)) 
                    for c, inc, u in values
                ])
                f.write(f" & {i*10}\% & {values_str} \\\\\n")
        f.write("\\end{tabular}\n")



def table2(ablation_type, seed):
    type_path = base_path + f"{ablation_type}/"
    save_path = base_save_path + f"{ablation_type}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(f"{save_path}/all_llms.txt", "w") as f:
        f.write("\\begin{tabular}{l|ccccc}\n")
        f.write("LLM & 1 & 2 & 3 & 4 & 5 \\\\\n")
        f.write("\\hline\n")
        for llm in llms:
            data = {}
            accuracies = []
            for i in range(1, 6):
                # load csv
                df = pd.read_csv(f"{type_path}{i}_{seed}/{llm}.csv")
                correct = 0
                incorrect = 0
                for dataset in datasets:
                    correct += df.loc[df['dataset'] == dataset, 'tagging_correct'].item()
                    incorrect += df.loc[df['dataset'] == dataset, 'tagging_incorrect'].item()
                accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else np.nan
                accuracies.append(accuracy)

            accuracies_str = " & ".join([f"{accuracy:.2f}" if not np.isnan(accuracy) else "-" for accuracy in accuracies])
            f.write(f"{llm} & {accuracies_str} \\\\\n")
        f.write("\\end{tabular}\n")

def plots(ablation_type, seed):
    llm_data = {}
    type_path = base_path + f"{ablation_type}/"
    save_path = base_save_path + f"{ablation_type}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for llm in llms:
        data = {}
        for i in range(1, 6):
            # load csv
            df = pd.read_csv(f"{type_path}{i}_{seed}/{llm}.csv")
            for dataset in datasets:
                correct = df.loc[df['dataset'] == dataset, 'tagging_correct'].sum()
                incorrect = df.loc[df['dataset'] == dataset, 'tagging_incorrect'].sum()
                accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else np.nan
                if dataset not in data:
                    data[dataset] = []
                data[dataset].append(accuracy)

        # make plots
        color_per_ds = {dataset: custom_palette[idx] for idx, dataset in enumerate(datasets)}
        for dataset, accuracies in data.items():
            y = np.array(accuracies, dtype=float)
            x = np.arange(1, 6)
            
            # Create a filled version for the dashed connections (NaN -> 0.5)
            y_filled = y.copy()
            nan_mask = np.isnan(y_filled)
            y_filled[nan_mask] = 0.5
            
            # 1. Plot the "skeleton" dashed line (connects everything)
            plt.plot(x, y_filled, color=color_per_ds[dataset], linestyle='--', alpha=0.6)
            
            # 2. Plot solid line for valid data (includes label for legend)
            plt.plot(x, y, marker='o', label=dataset_names[dataset], color=color_per_ds[dataset], linestyle='-')
            
            # 3. Plot small markers for missing data at 0.5
            if np.any(nan_mask):
                plt.scatter(x[nan_mask], y_filled[nan_mask], color=color_per_ds[dataset], s=15, zorder=3)

        plt.xlabel('Number Edges')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Number Edges')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/{llm}.pdf")
        plt.clf()

        llm_data[llm] = data

    # Calculate average accuracy over all LLMs
    average_data = {dataset: [] for dataset in datasets}

    for dataset in datasets:
        for i in range(5):
            accuracies = [llm_data[llm][dataset][i] for llm in llms if dataset in llm_data[llm]]
            average_accuracy = np.nanmean(accuracies)
            average_data[dataset].append(average_accuracy)

    # Plot average accuracy
    for dataset, accuracies in average_data.items():
        plt.plot(range(1, 6), accuracies, marker='o', label=dataset_names[dataset], color=color_per_ds[dataset])

    plt.plot(range(1, 6), [np.nanmean([average_data[dataset][i] for dataset in datasets]) for i in range(5)], marker='o', label='Average')

    plt.xlabel('Undirect Number')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracy vs Undirect Number')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/average.pdf")
    plt.clf()


def combined_remove_inverse(llm, seed):
    remove_data = {}
    inverse_data = {}
    
    for ablation_type, data_dict in zip(["remove", "inverse"], [remove_data, inverse_data]):
        type_path = base_path + f"{ablation_type}/"
        data = {}
        for i in range(1, 6):
            # load csv
            df = pd.read_csv(f"{type_path}{i}_{seed}/{llm}.csv")
            for dataset in datasets:
                correct = df.loc[df['dataset'] == dataset, 'tagging_correct'].sum()
                incorrect = df.loc[df['dataset'] == dataset, 'tagging_incorrect'].sum()
                accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else np.nan
                if dataset not in data:
                    data[dataset] = []
                data[dataset].append(accuracy)
        data_dict[llm] = data

    # Plot side by side with shared y-axis
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X']
    for idx, dataset in enumerate(datasets):
        marker = markers[idx % len(markers)]
        markersize = 17 if marker == '*' else 13
        color = custom_palette[idx]
        
        # Determine zorder based on your original logic
        z_order = 10 if dataset == "bnlearn_earthquake" else 2
        
        # Loop over both axes (remove_data and inverse_data)
        for ax, d_source in zip(axes, [remove_data, inverse_data]):
            y = np.array(d_source[llm][dataset], dtype=float)
            x = np.arange(1, 6)
            
            # Create filled version for dashed line (NaN -> 0.5)
            y_filled = y.copy()
            nan_mask = np.isnan(y_filled)
            y_filled[nan_mask] = 0.5
            
            # 1. Dashed connection line (using imputed data)
            ax.plot(x, y_filled, color=color, linestyle='--', linewidth=3, zorder=z_order)
            
            # 2. Solid line + Main Markers (Valid data only)
            # Note: We only add the label if it's the first axis (axes[0]) or handle duplicates later
            lbl = dataset_names[dataset] 
            ax.plot(x, y, marker=marker, label=lbl, markersize=markersize, 
                    linewidth=3, color=color, linestyle='-', zorder=z_order)
            
            # 3. Small markers for missing data
            if np.any(nan_mask):
                # Calculate smaller size (approx half the diameter of the main marker)
                s_size = (markersize * 0.5) ** 2 
                ax.scatter(x[nan_mask], y_filled[nan_mask], marker=marker, 
                           s=s_size, color=color, zorder=z_order)

    # axes[0].set_title('Fewer Edges', fontsize=34)
    axes[0].set_xlabel('Removed Edges', fontsize=30)
    axes[0].set_ylabel('Accuracy', fontsize=30)
    axes[0].tick_params(axis='both', which='major', labelsize=24)
    axes[0].set_xticks(range(1, 6))
    axes[0].set_xticklabels([f'{i*10}%' for i in range(1, 6)])
    axes[0].set_ylim(-0.05, 1.05)  # Set y-axis limits to ensure consistency
    axes[0].grid(True)

    # axes[1].set_title('Incorrect Edges', fontsize=34)
    axes[1].set_xlabel('Inverted Edges', fontsize=30)
    axes[1].tick_params(axis='both', which='major', labelsize=24)
    axes[1].set_xticks(range(1, 6))
    axes[1].set_xticklabels([f'{i*10}%' for i in range(1, 6)])
    axes[1].set_ylim(-0.05, 1.05)  # Set y-axis limits to ensure consistency
    axes[1].grid(True)

    # Create a single legend for both plots with markers
    handles, labels = axes[0].get_legend_handles_labels()
    scatter_handles = []
    for handle in handles:
        marker = handle.get_marker()
        markersize = 35 if marker == '*' else 25
        scatter_handles.append(plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=handle.get_color(), markersize=markersize))
    
    # Reduce the white space between marker and text
    # fig.legend(scatter_handles[:6], labels[:6], loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.1), fontsize=22, handletextpad=0.2)
    # fig.legend(scatter_handles[6:], labels[6:], loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.2), fontsize=22, handletextpad=0.2)

    fig.legend(scatter_handles[:4], labels[:4], loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.11), fontsize=26, handletextpad=0.2)
    fig.legend(scatter_handles[4:8], labels[4:8], loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.21), fontsize=26, handletextpad=0.2)
    fig.legend(scatter_handles[8:], labels[8:], loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.31), fontsize=26, handletextpad=0.2)

    plt.tight_layout()
    plt.savefig(f"{base_save_path}/{llm}_combined_remove_inverse.pdf", bbox_inches='tight')
    plt.clf()


base_path = "results/main_evaluation/_ablation/"
base_save_path = "plots/ablation/"
datasets = [
    "bnlearn_cancer", "bnlearn_earthquake", "bnlearn_survey", 
    "bnlearn_asia", "lucas", "bnlearn_child", 
    "bnlearn_insurance", "bnlearn_alarm", "bnlearn_hailfinder", "bnlearn_hepar2", "bnlearn_win95pts"
]
dataset_names = {"bnlearn_child": "Child", "bnlearn_earthquake": "Earthquake", "bnlearn_insurance": "Insurance",
                    "bnlearn_survey": "Survey", "bnlearn_asia": "Asia", "bnlearn_cancer": "Cancer",
                    "bnlearn_alarm": "Alarm", "lucas": "Lucas", "bnlearn_hepar2": "Hepar2", "bnlearn_win95pts": "Win95Pts", "bnlearn_hailfinder": "Hailfinder"}

llms = ["openai--gpt-5.2",
        "anthropic--claude-opus-4.6",
        "google--gemini-3-pro-preview",
        "meta-llama--llama-3.3-70b-instruct",
        "qwen--qwen3.5-397b-a17b",
        "z-ai--glm-5",
        "minimax--minimax-m2.5"
]
llm_to_text = {"openai--gpt-5.2": "GPT 5.2", "anthropic--claude-opus-4.6": "Claude 4.6", "google--gemini-3-pro-preview": "Gemini 3 Pro", "meta-llama--llama-3.3-70b-instruct": "Llama 3.3", "qwen--qwen3.5-397b-a17b": "Qwen 3.5", "z-ai--glm-5": "GLM 5", "minimax--minimax-m2.5": "Minimax 2.5"}
seed=0

for ablation_type in ["undirect", "remove", "inverse"]:
    table(ablation_type, seed=seed)
    table2(ablation_type, seed=seed)
    plots(ablation_type, seed=seed)

for llm in llms:
    combined_remove_inverse(llm, seed=seed)   
