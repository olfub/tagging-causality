import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

custom_palette = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', 
                '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC', '#493267']

def plots(ablation_type):
    llm_data = {}
    llm_abstained_data = {}
    type_path = base_path + f"{ablation_type}/"
    save_path = base_save_path + f"{ablation_type}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for llm in llms:
        data = {}
        abstained_data = {}
        for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            data_seed = {}
            for seed in seeds:
                # load csv
                df = pd.read_csv(f"{type_path}1_{seed}_{i}/{llm}.csv")
                for dataset in datasets:
                    correct = df.loc[df['dataset'] == dataset, 'tagging_correct'].sum()
                    incorrect = df.loc[df['dataset'] == dataset, 'tagging_incorrect'].sum()
                    undirected = df.loc[df['dataset'] == dataset, 'tagging_nothing'].sum()
                    if dataset not in data_seed:
                        data_seed[dataset] = []
                    data_seed[dataset].append((correct, incorrect, undirected))
            for dataset in datasets:
                if dataset not in data:
                    data[dataset] = []
                if dataset not in abstained_data:
                    abstained_data[dataset] = []
                average_correct = np.mean([x[0] for x in data_seed[dataset]])
                average_incorrect = np.mean([x[1] for x in data_seed[dataset]])
                average_undirected = np.mean([x[2] for x in data_seed[dataset]])
                directed_total = average_correct + average_incorrect
                all_total = directed_total + average_undirected
                data[dataset].append(
                    average_correct / directed_total if directed_total > 0 else np.nan
                )
                abstained_data[dataset].append(
                    average_undirected / all_total if all_total > 0 else np.nan
                )

        # make plots

        # filter out datasets with only NaN values
        filtered_data = {dataset: accuracies for dataset, accuracies in data.items() if not all(np.isnan(accuracies))}
        filtered_legend = [dataset_names[dataset] for dataset in filtered_data.keys()]

        plt.figure(figsize=(16, 7))
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X']
        offset_scale = 0.005  # Adjust this if lines are still too close or too far
        default_colors = custom_palette
        for idx, dataset in enumerate(datasets):
            if dataset not in filtered_data:
                continue	
            accuracies = data[dataset]
            color = default_colors[idx % len(default_colors)]
            marker = markers[idx % len(markers)]
            markersize = 20 if marker == '*' else 15
            y = np.array(accuracies, dtype=float)
            x_vals = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            
            # Create filled version for dashed line (NaN -> 0.5)
            y_filled = y.copy()
            nan_mask = np.isnan(y_filled)
            # Apply offset ONLY to the filled values (the dashed parts)
            # We use (idx - len(datasets)/2) to center the spread around 0.5
            jitter = (idx - len(datasets) / 2) * offset_scale
            y_filled[nan_mask] = 0.5 + jitter
            
            # 1. Dashed line: explicit color, NO legend label
            plt.plot(x_vals, y_filled, color=color, linestyle='--', linewidth=5, zorder=9, alpha=0.6, label='_nolegend_')
            
            # 2. Solid line: explicit color, KEEP the label here
            plt.plot(x_vals, y, marker=marker, label=dataset_names[dataset], 
                     markersize=markersize, linewidth=5, zorder=10, color=color, linestyle='-')
            
            # 3. Scatter: explicit color, NO legend label
            if np.any(nan_mask):
                plt.scatter(x_vals[nan_mask], y_filled[nan_mask], color=color, s=60, zorder=10, label='_nolegend_')
            plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])

        all_abstained = [np.nanmean([abstained_data[dataset][i] for dataset in datasets]) for i in range(10)]
        plt.plot(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            all_abstained,
            linestyle=':',
            linewidth=6,
            color='black',
            alpha=0.8,
            label='% abstained'
        )

        plt.xlabel('Tag Error Percentage', fontsize=30)
        plt.ylabel('Accuracy', fontsize=30)
        plt.ylim(-0.05, 1.05)  # Set y-axis limits to ensure consistency
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.grid(True)
        
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=26)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{llm}.pdf")
        plt.clf()

        llm_data[llm] = data
        llm_abstained_data[llm] = abstained_data

        # plot undirected percentages per dataset for this LLM
        filtered_data = {
            dataset: values
            for dataset, values in abstained_data.items()
            if not all(np.isnan(values))
        }

        plt.figure(figsize=(16, 7))
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X']
        default_colors = custom_palette
        for idx, dataset in enumerate(datasets):
            if dataset not in filtered_data:
                continue
            abstained = abstained_data[dataset]
            color = default_colors[idx % len(default_colors)]
            marker = markers[idx % len(markers)]
            markersize = 20 if marker == '*' else 15
            y = np.array(abstained, dtype=float)
            x_vals = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

            plt.plot(x_vals, y, marker=marker, label=dataset_names[dataset], 
                     markersize=markersize, linewidth=5, zorder=10, color=color, linestyle='-')

            plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])

        plt.xlabel('Tag Error Percentage', fontsize=30)
        plt.ylabel('Undirected Percentage', fontsize=30)
        plt.ylim(-0.05, 1.05)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=26)
        plt.tight_layout()
        plt.savefig(f"{save_path}/_undirected_{llm}.pdf")
        plt.clf()

        llm_abstained_data[llm] = abstained_data

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

plots("tags")