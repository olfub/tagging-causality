import os
import csv
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from discovery.pc_tagged import get_edge_count
from util import load_data, load_from_llm, make_deterministic

datasets = [
    "bnlearn_cancer", "bnlearn_earthquake", "bnlearn_survey", 
    "bnlearn_asia", "lucas", "bnlearn_child", 
    "bnlearn_insurance", "bnlearn_alarm", "bnlearn_hailfinder", "bnlearn_hepar2", "bnlearn_win95pts"
]
llms = ["openai--gpt-5.2",
        "anthropic--claude-opus-4.6",
        "google--gemini-3-pro-preview",
        "meta-llama--llama-3.3-70b-instruct",
        "qwen--qwen3.5-397b-a17b",
        "z-ai--glm-5",
        "minimax--minimax-m2.5"
]
llm_to_text = {"openai--gpt-5.2": "GPT 5.2", "anthropic--claude-opus-4.6": "Claude 4.6", "google--gemini-3-pro-preview": "Gemini 3 Pro", "meta-llama--llama-3.3-70b-instruct": "Llama 3.3", "qwen--qwen3.5-397b-a17b": "Qwen 3.5", "z-ai--glm-5": "GLM 5", "minimax--minimax-m2.5": "Minimax 2.5"}
llm_seed = 0

dataset_names = {"bnlearn_child": "Child", "bnlearn_earthquake": "Earthquake", "bnlearn_insurance": "Insurance",
                 "bnlearn_survey": "Survey", "bnlearn_asia": "Asia", "bnlearn_cancer": "Cancer",
                 "bnlearn_alarm": "Alarm", "lucas": "Lucas", "bnlearn_hepar2": "Hepar2", "bnlearn_win95pts": "Win95Pts", "bnlearn_hailfinder": "Hailfinder"}

experimental_series = "main_evaluation"
config = json.load(open(f"results/{experimental_series}/_configs/_best_config_tag_pc_0_on_ges.json", "r"))
order_data = config["order_data"]
best_llm = config["llm"]
nr_samples = config["nr_samples"]
pc_indep_test = config["pc_indep_test"]
pc_alpha = config["pc_alpha"]
ges_indep_test = config["ges_indep_test"]
min_samples = config["min_samples"]
compute_min_prob_threshold = config["compute_min_prob_threshold"]
min_prob_threshold_default = config["min_prob_threshold_default"]
anti_tags = config["anti_tags"]
remove_duplicates = config["remove_duplicates"]
remove_singular_tags = config["remove_singular_tags"]
prior_on_weight = config["prior_on_weight"]
always_meeks = config["always_meeks"]
redirect_existing_edges = config["redirect_existing_edges"]
redirecting_strategy = config["redirecting_strategy"]
min_prob_redirecting = config["min_prob_redirecting"]
include_current_edge_as_evidence = config["include_current_edge_as_evidence"]
include_redirected_edges_in_edge_count = config["include_redirected_edges_in_edge_count"]

min_evidence = 1
min_samples_relations = 3
max_tag_pairs = 7

accuracies = {}

make_deterministic(0)
if os.path.exists(f'plots/homogeneity/{best_llm}_pairs.txt'):
    os.remove(f'plots/homogeneity/{best_llm}_pairs.txt')

# Define the shared colormap
colors = ["lightgrey", "dodgerblue"]
cmap = LinearSegmentedColormap.from_list("lightgrey_dodgerblue", colors, N=256)
cmap.set_bad(color='white')

# Set up the large grid figure
fig_grid, axes_grid = plt.subplots(nrows=len(datasets), ncols=len(llms), figsize=(40, 55))
plt.subplots_adjust(left=0.08, right=0.9, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

# create heatmap for each dataset and llm
for c, llm in enumerate(llms):
    for r, dataset in enumerate(datasets):
        
        # load data (graph)
        variables, var_labels, _, edges, positions, _ = load_data(dataset, nr_samples, order_data=order_data, seed=0)

        # load tags
        data_id = dataset.split("_")[-1] if dataset.startswith("bnlearn") else dataset
        tags = load_from_llm(f"{llm}__tag__{data_id}___{llm_seed}.txt", variables=var_labels, anti_tags=anti_tags, remove_duplicates=remove_duplicates, remove_singular_tags=remove_singular_tags)
        tag_list = [tags[var] for var in var_labels]

        # compute edge count
        unique_tags = list(set([tag for var_tags in tag_list for tag in var_tags]))
        unique_tags.sort()
        edge_count = get_edge_count(unique_tags, tags, edges)

        # edge_count to matrix
        edge_count_matrix = np.zeros((len(unique_tags), len(unique_tags)), dtype=int)
        for edge, count in edge_count.items():
            edge_0 = unique_tags.index(edge[0])
            edge_1 = unique_tags.index(edge[1])
            edge_count_matrix[edge_0][edge_1] = count

        # calculate accuracy
        edge_acc_matrix = np.zeros_like(edge_count_matrix, dtype=float)
        evidence_ones = np.zeros_like(edge_count_matrix, dtype=int)
        evidence_twos = np.zeros_like(edge_count_matrix, dtype=int)
        for i in range(len(unique_tags)):
            for j in range(len(unique_tags)):
                if i < j:
                    continue
                evidence_sum = edge_count_matrix[i][j] + edge_count_matrix[j][i]
                if evidence_sum == 1:
                    evidence_ones[i][j] = 1
                    evidence_ones[j][i] = 1
                elif evidence_sum == 2:
                    evidence_twos[i][j] = 1
                    evidence_twos[j][i] = 1
                if evidence_sum < min_evidence:
                    edge_acc_matrix[i][j] = -1
                    edge_acc_matrix[j][i] = -1
                    continue
                acc = edge_count_matrix[i][j] / evidence_sum
                if acc < 0.5:
                    acc = 1 - acc
                edge_acc_matrix[i][j] = acc
                edge_acc_matrix[j][i] = acc

        # normalize to [0, 1]
        edge_acc_matrix = (edge_acc_matrix - 0.5) * 2
        edge_acc_matrix[edge_acc_matrix < 0] = np.nan
        edge_acc_matrix_masked = np.ma.masked_invalid(edge_acc_matrix)

        # Determine hatch settings
        if len(unique_tags) < 10:
            hatch_lw = 6
            one_hatch = 'X'
            two_hatch = '/'
        elif len(unique_tags) < 20:
            hatch_lw = 3
            one_hatch = 'XX'
            two_hatch = '//'
        else:
            hatch_lw = 1
            one_hatch = 'XXXX'
            two_hatch = '////'
        print(f"LLM: {llm}, Dataset: {dataset}, heatmap size: {len(unique_tags)}")
        # --- Helper function to draw identical plots on any axis ---
        def apply_heatmap_styling(target_ax, draw_cbar):
            sns.heatmap(edge_acc_matrix_masked, ax=target_ax, xticklabels=False, yticklabels=False, cmap=cmap, annot=False, vmin=0, vmax=1, cbar=draw_cbar)

            # Add a box around all heatmap elements that are not NaN
            for i in range(len(unique_tags)):
                for j in range(len(unique_tags)):
                    if not np.isnan(edge_acc_matrix[i, j]):
                        if len(edge_acc_matrix) < 40:  # If the heatmap is small, use a line width for better visibility
                            lw = 1
                        else:
                            lw = 0
                        target_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=lw, zorder=10))
            # Add hatches
            for i in range(len(unique_tags)):
                for j in range(len(unique_tags)):
                    if evidence_ones[i, j] == 1:
                        target_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch=one_hatch, edgecolor='white', lw=0))
                    elif evidence_twos[i, j] == 1:
                        target_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch=two_hatch, edgecolor='white', lw=0))

        # ==========================================
        # 1. DRAW AND SAVE INDIVIDUAL PLOT
        # ==========================================
        fig_indiv, ax_indiv = plt.subplots(figsize=(10, 8))
        
        # Use context manager to ensure proper hatch linewidth for the individual plot
        with mpl.rc_context({'hatch.linewidth': hatch_lw}):
            apply_heatmap_styling(ax_indiv, draw_cbar=True)
            
            # Format the individual colorbar exactly as you had it
            cbar = ax_indiv.collections[0].colorbar
            if cbar:
                cbar.ax.tick_params(labelsize=34, length=20, width=4)
                
            plt.xticks(rotation=45, ha='right')
            ax_indiv.set_title(dataset_names.get(dataset, dataset), fontsize=16)
            plt.tight_layout()
            
            # Save and immediately close to free memory
            plt.savefig(f"plots/homogeneity/{llm}_{dataset}_tag_homogenity.pdf")
        plt.close(fig_indiv)

        # ==========================================
        # 2. DRAW ON THE COMBINED GRID
        # ==========================================
        ax_grid = axes_grid[r, c]
        
        # For the grid, we don't draw individual colorbars
        apply_heatmap_styling(ax_grid, draw_cbar=False)

        # Add Column Titles (LLM Names) to the top row
        if r == 0:
            llm_str = llm_to_text[llm]
            ax_grid.set_title(llm_str, fontsize=30, pad=20, fontweight="bold")
            
        # Add Row Titles (Dataset Names) to the first column
        if c == 0:
            ax_grid.set_ylabel(dataset_names.get(dataset, dataset), fontsize=30, labelpad=20, fontweight="bold")

        # save accuracy for CSV/TXT processing later
        avg_accuracy = np.nanmean(edge_acc_matrix)
        accuracies[(llm, dataset)] = avg_accuracy

        # --- Text file processing logic ---
        if llm == best_llm:
            with open(f'plots/homogeneity/{best_llm}_pairs.txt', 'a') as f:
                f.write(f" & \\textbf{{{dataset_names[dataset]}}} & & \\nonumber \\\\\n")
                indices = np.argwhere(edge_acc_matrix > 0)
                edges_and_evidences = []
                already_done = set()
                for index in indices:
                    if edge_count_matrix[index[0], index[1]] + edge_count_matrix[index[1], index[0]] >= min_samples_relations:
                        if (index[1], index[0]) in already_done:
                            continue
                        if edge_count_matrix[index[0], index[1]] > edge_count_matrix[index[1], index[0]]:
                            edges_and_evidences.append((index[0], index[1], edge_count_matrix[index[0], index[1]], edge_count_matrix[index[1], index[0]]))
                        else:
                            edges_and_evidences.append((index[1], index[0], edge_count_matrix[index[1], index[0]], edge_count_matrix[index[0], index[1]]))
                    already_done.add((index[0], index[1]))
                edges_and_evidences.sort(key=lambda x: (x[2] / (x[2] + x[3]), x[2] + x[3]), reverse=True)
                edges_and_evidences = edges_and_evidences[:max_tag_pairs]
                for edge in edges_and_evidences:
                    acc = int(100 * edge[2] / (edge[2] + edge[3]))
                    start = unique_tags[edge[0]].replace("_", "\\_")
                    end = unique_tags[edge[1]].replace("_", "\\_")
                    f.write(f"\\text{{``{start}''}} &~\\rightarrow~ \\text{{``{end}''}} && ({acc}\% ~/~ {edge[2] + edge[3]}~~) \\nonumber \\\\\n")


# ==========================================
# 3. FINALIZE AND SAVE THE COMBINED GRID
# ==========================================
# Add Single Shared Colorbar to the far right
cbar_ax = fig_grid.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
cbar = fig_grid.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=34, length=20, width=4)

# Save the final combined figure
# Note: matplotlib saves the figure using the last active rcParams state. 
# Hatch line widths in the big grid might default to the final iteration's setting.
fig_grid.savefig("plots/homogeneity/combined_homogeneity_grid.pdf", bbox_inches='tight')
print("Successfully generated all individual plots and combined_homogeneity_grid.pdf")
plt.close(fig_grid)

# ==========================================
# 4. DRAW SPECIAL ROW FOR BEST LLM (Earthquake, Child, Lucas, Alarm, Insurance, Hailfinder)
# ==========================================
special_datasets = ["bnlearn_earthquake", "bnlearn_child", "lucas", "bnlearn_alarm", "bnlearn_insurance", "bnlearn_hailfinder"]

# Setup figure for 1 row, 6 columns
fig_special, axes_special = plt.subplots(nrows=1, ncols=6, figsize=(30, 5))
# Adjust subplots to make room on the right for the colorbar
plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)

# Ensure we use the best_llm specifically
target_llm = best_llm 

print(f"Generating special row for LLM: {target_llm}")

for i, dataset in enumerate(special_datasets):
    ax = axes_special[i]
    
    # --- REPEAT DATA LOADING & MATRIX CALCULATION ---
    # (Re-calculating specifically for this row to ensure independence from the main loop state)
    variables, var_labels, _, edges, positions, _ = load_data(dataset, nr_samples, order_data=order_data, seed=0)
    data_id = dataset.split("_")[-1] if dataset.startswith("bnlearn") else dataset
    tags = load_from_llm(f"{target_llm}__tag__{data_id}___{llm_seed}.txt", variables=var_labels, anti_tags=anti_tags, remove_duplicates=remove_duplicates, remove_singular_tags=remove_singular_tags)
    tag_list = [tags[var] for var in var_labels]

    unique_tags = list(set([tag for var_tags in tag_list for tag in var_tags]))
    unique_tags.sort()
    edge_count = get_edge_count(unique_tags, tags, edges)

    edge_count_matrix = np.zeros((len(unique_tags), len(unique_tags)), dtype=int)
    for edge, count in edge_count.items():
        edge_0 = unique_tags.index(edge[0])
        edge_1 = unique_tags.index(edge[1])
        edge_count_matrix[edge_0][edge_1] = count

    edge_acc_matrix = np.zeros_like(edge_count_matrix, dtype=float)
    evidence_ones = np.zeros_like(edge_count_matrix, dtype=int)
    evidence_twos = np.zeros_like(edge_count_matrix, dtype=int)

    for r_idx in range(len(unique_tags)):
        for c_idx in range(len(unique_tags)):
            if r_idx < c_idx: continue
            evidence_sum = edge_count_matrix[r_idx][c_idx] + edge_count_matrix[c_idx][r_idx]
            if evidence_sum == 1:
                evidence_ones[r_idx][c_idx] = 1; evidence_ones[c_idx][r_idx] = 1
            elif evidence_sum == 2:
                evidence_twos[r_idx][c_idx] = 1; evidence_twos[c_idx][r_idx] = 1
            if evidence_sum < min_evidence:
                edge_acc_matrix[r_idx][c_idx] = -1; edge_acc_matrix[c_idx][r_idx] = -1
                continue
            acc = edge_count_matrix[r_idx][c_idx] / evidence_sum
            if acc < 0.5: acc = 1 - acc
            edge_acc_matrix[r_idx][c_idx] = acc; edge_acc_matrix[c_idx][r_idx] = acc

    edge_acc_matrix = (edge_acc_matrix - 0.5) * 2
    edge_acc_matrix[edge_acc_matrix < 0] = np.nan
    edge_acc_matrix_masked = np.ma.masked_invalid(edge_acc_matrix)

    # --- DETERMINE HATCH SETTINGS ---
    if len(unique_tags) < 10:
        hatch_lw = 6; one_hatch = 'X'; two_hatch = '/'
    elif len(unique_tags) < 20:
        hatch_lw = 3; one_hatch = 'XX'; two_hatch = '//'
    else:
        hatch_lw = 1; one_hatch = 'XXXX'; two_hatch = '////'

    # --- PLOTTING ---
    sns.heatmap(edge_acc_matrix_masked, ax=ax, xticklabels=False, yticklabels=False, cmap=cmap, annot=False, vmin=0, vmax=1, cbar=False)

    # Add boxes
    for r_idx in range(len(unique_tags)):
        for c_idx in range(len(unique_tags)):
            if not np.isnan(edge_acc_matrix[r_idx, c_idx]):
                lw = 1 if len(edge_acc_matrix) < 40 else 0
                ax.add_patch(plt.Rectangle((c_idx, r_idx), 1, 1, fill=False, edgecolor='black', lw=lw, zorder=10))
            
            # Add hatches
            if evidence_ones[r_idx, c_idx] == 1:
                ax.add_patch(plt.Rectangle((c_idx, r_idx), 1, 1, fill=False, hatch=one_hatch, edgecolor='white', lw=0))
            elif evidence_twos[r_idx, c_idx] == 1:
                ax.add_patch(plt.Rectangle((c_idx, r_idx), 1, 1, fill=False, hatch=two_hatch, edgecolor='white', lw=0))

    ax.set_title(dataset_names.get(dataset, dataset), fontsize=24)

# --- ADD COLORBAR TO SELECTED ROW PLOT ---
# Position: [left, bottom, width, height] in figure coordinates
cbar_ax = fig_special.add_axes([0.92, 0.15, 0.01, 0.7]) 
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
cbar = fig_special.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=20)

# Save the special row figure
fig_special.savefig(f"plots/homogeneity/{target_llm}_selected_row.pdf", bbox_inches='tight')
plt.close(fig_special)
# Write homogenity table to csv file
with open('plots/homogeneity/homogenity.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # write header
    header = ["LLM"] + datasets + ["Average"]
    writer.writerow(header)
    # write data
    for llm in llms:
        avg_accuracy = np.nanmean([accuracies[(llm, dataset)] for dataset in datasets])
        row = [llm] + [f"{accuracies[(llm, dataset)]:.4f}" for dataset in datasets] + [f"{avg_accuracy:.4f}"]
        writer.writerow(row)
    # average per dataset
    avg_accuracies = [sum([accuracies[(llm, dataset)] for llm in llms]) / len(llms) for dataset in datasets]
    writer.writerow(["Average"] + [f"{np.nanmean([accuracies[(llm, dataset)] for llm in llms]):.4f}" for dataset in datasets] + [f"{np.nanmean(avg_accuracies):.4f}"])