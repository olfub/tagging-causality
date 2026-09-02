import os

import numpy as np
from scipy.stats import rankdata

# must mirror E6_llm_model_versions.sh
experimental_series = "E6_llm_model_versions"
identifiers = ["bnlearn_child", "bnlearn_earthquake", "bnlearn_insurance", "bnlearn_survey", "bnlearn_asia",
               "bnlearn_cancer", "bnlearn_alarm", "lucas", "bnlearn_hepar2", "bnlearn_win95pts", "bnlearn_hailfinder"]
order_data = "random"
nr_samples = 10000
seeds = list(range(10))
pc_indep_test = "chisq"
pc_alpha = 0.05
ges_score_func = "local_score_BDeu"

# shared across all three LLM blocks in E6_llm_model_versions.sh
anti_tags = False
remove_duplicates = True
remove_singular_tags = False
min_prob_threshold_default = 0.5
fewer_tags_str = "00"
include_redirected_edges_in_edge_count = True
min_prob_redirecting = 0.6

methods_to_consider = ["tag_pc_1", "tag_pc_0", "tag_pc_0_on_skel_v", "tag_pc_0_on_ges"]

# one group per LLM block in E6_llm_model_versions.sh, with that block's fixed config
llm_groups = [
    {
        "llms": ["anthropic--claude-opus-5", "anthropic--claude-opus-4.8", "anthropic--claude-opus-4.7", "anthropic--claude-opus-4.6"],
        "min_samples": 2, "compute_min_prob_threshold": True, "prior_on_weight": False,
        "always_meeks": False, "redirect_existing_edges": True, "redirecting_strategy": 0,
        "include_current_edge_as_evidence": True,
    },
    {
        "llms": ["openai--gpt-5.6-sol", "openai--gpt-5.5", "openai--gpt-5.4", "openai--gpt-5.2"],
        "min_samples": 1, "compute_min_prob_threshold": True, "prior_on_weight": True,
        "always_meeks": True, "redirect_existing_edges": True, "redirecting_strategy": 1,
        "include_current_edge_as_evidence": True,
    },
    {
        "llms": ["qwen--qwen3.8-max", "qwen--qwen3.7-max", "qwen--qwen3.6-max-preview", "qwen--qwen3.5-397b-a17b"],
        "min_samples": 1, "compute_min_prob_threshold": True, "prior_on_weight": True,
        "always_meeks": True, "redirect_existing_edges": True, "redirecting_strategy": 1,
        "include_current_edge_as_evidence": True,
    },
]

llm_to_text = {
    "anthropic--claude-opus-5": "Claude Opus 5",
    "anthropic--claude-opus-4.8": "Claude Opus 4.8",
    "anthropic--claude-opus-4.7": "Claude Opus 4.7",
    "anthropic--claude-opus-4.6": "Claude Opus 4.6",
    "openai--gpt-5.6-sol": "GPT 5.6 Sol",
    "openai--gpt-5.5": "GPT 5.5",
    "openai--gpt-5.4": "GPT 5.4",
    "openai--gpt-5.2": "GPT 5.2",
    "qwen--qwen3.8-max": "Qwen3 8 Max",
    "qwen--qwen3.7-max": "Qwen3 7 Max",
    "qwen--qwen3.6-max-preview": "Qwen3 6 Max Preview",
    "qwen--qwen3.5-397b-a17b": "Qwen3 5 397B",
}

method_names = {
    "tag_pc_1": "Tagged-PC (AntiV)",
    "tag_pc_0": "Tagged-PC",
    "tag_pc_0_on_skel_v": "Tagging on GT CPDAG",
    "tag_pc_0_on_ges": "Tagged-GES",
}


def safe_filename_label(label):
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in label)


def read_eval_csv(file):
    with open(file, "r") as f:
        metrics, models = [], []
        for line in f.read().splitlines():
            parts = line.split(",")
            models.append(parts[0])
            metrics.append(parts[1:])
        return np.array(metrics, dtype=float), models


def eval_csv_path(llm, config, identifier, seed):
    path = f"results/{experimental_series}/{identifier}/{order_data}/{seed}/{pc_indep_test}_{pc_alpha}_{ges_score_func}_{nr_samples}"
    path_llm = f"{path}/{llm}"
    path_tagging = (
        f"{path_llm}/{anti_tags}_{remove_duplicates}_{remove_singular_tags}_{config['prior_on_weight']}_"
        f"{config['min_samples']}_{config['compute_min_prob_threshold']}_{min_prob_threshold_default}_{fewer_tags_str}"
    )
    path_tagging_alg0 = (
        f"{path_tagging}/{config['always_meeks']}_{config['redirect_existing_edges']}_{config['redirecting_strategy']}_"
        f"{config['include_current_edge_as_evidence']}_{include_redirected_edges_in_edge_count}_{min_prob_redirecting}"
    )
    return f"{path_tagging_alg0}/_tagging_alg0/eval.csv"


def read_llm_results(llm, config):
    # returns dict[(identifier, seed)] -> (n_methods, 7) eval matrix, and the model order
    results = {}
    models = None
    for identifier in identifiers:
        for seed in seeds:
            eval_file = eval_csv_path(llm, config, identifier, seed)
            eval_result, file_models = read_eval_csv(eval_file)
            if models is None:
                models = file_models
            else:
                assert models == file_models, f"Inconsistent methods in {eval_file}"
            if identifier in ["bnlearn_hepar2", "bnlearn_win95pts", "bnlearn_hailfinder"]:
                # SID is not computed for these (too large), stored as 0 -> mask out
                assert np.all(eval_result[:, 2:4] == 0)
                eval_result[:, 2:4] = np.nan
            results[(identifier, seed)] = eval_result
    return results, models


def metrics_to_ranks(metrics):
    # metrics: (n_llms, 7); first 4 columns minimized (SHD/SID), last 3 maximized (P/R/F1)
    ranks = np.zeros_like(metrics)
    for i in range(metrics.shape[1]):
        if i < 4:
            ranks[:, i] = rankdata(metrics[:, i], method="average")
        else:
            ranks[:, i] = rankdata(-metrics[:, i], method="average")
    return ranks


def average_family_ranks(family_llm_results, method):
    # family_llm_results: list of (results, models) for the LLMs in one family (e.g. the 4 Claude versions)
    # for every dataset+seed, rank these LLMs against each other (never against another family),
    # average those ranks over the datasets within one seed, then report mean and std over the seeds
    # (same convention as results_eval.py: the std is the spread of the per-seed average ranks)
    method_indices = [models.index(method) for _, models in family_llm_results]
    per_seed_average_ranks = []
    for seed in seeds:
        per_ds_ranks = []
        for identifier in identifiers:
            rows = [results[(identifier, seed)][midx] for (results, _), midx in zip(family_llm_results, method_indices)]
            per_ds_ranks.append(metrics_to_ranks(np.array(rows)))
        per_seed_average_ranks.append(np.nanmean(np.array(per_ds_ranks), axis=0))
    per_seed_average_ranks = np.array(per_seed_average_ranks)
    return np.nanmean(per_seed_average_ranks, axis=0), np.nanstd(per_seed_average_ranks, axis=0)


def metrics_to_latex(file_path, data, row_labels, stds=None, row_groups=None, hlines_after_rows=None):
    # data: (n_rows, 7) of average ranks; lower rank is always better, bold the min per family group per column
    if row_groups is None:
        row_groups = [list(range(data.shape[0]))]

    bf_values = np.full((len(row_groups), data.shape[1]), np.nan)
    row_to_group = {}
    for g, indices in enumerate(row_groups):
        for j in range(data.shape[1]):
            col = data[indices, j]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                continue
            bf_values[g, j] = np.min(valid)
        for i in indices:
            row_to_group[i] = g

    with open(file_path, "w") as f:
        f.write("\\begin{tabular}{l|ccccccc}\n")
        f.write("LLM & SHD & SHD\\textsubscript{double} & SID\\textsubscript{min} & SID\\textsubscript{max} & Precision & Recall & F\\textsubscript{1} \\\\\n")
        f.write(" & Ranks & Ranks & Ranks & Ranks & Ranks & Ranks & Ranks \\\\\n")
        f.write("\\hline\n")
        for i, label in enumerate(row_labels):
            group_bf_values = bf_values[row_to_group[i]]
            f.write(f"{label} & ")
            for j in range(data.shape[1]):
                value = data[i, j]
                if np.isnan(value):
                    cell = "-"
                else:
                    cell = "$"
                    cell += f"\\mathbf{{{value:.2f}}}" if value == group_bf_values[j] else f"{value:.2f}"
                    if stds is not None:
                        cell += f" {{\\scriptstyle \\pm {stds[i, j]:.2f}}}"
                    cell += "$"
                sep = " \\\\\n" if j == data.shape[1] - 1 else " & "
                f.write(cell + sep)
            if hlines_after_rows is not None and i in hlines_after_rows:
                f.write("\\hline\n")
        f.write("\\end{tabular}\n")


def main():
    output_path = f"results/{experimental_series}/_result_tables"
    os.makedirs(output_path, exist_ok=True)

    # read raw per-(dataset, seed) results for every LLM once, reused across all 4 methods
    row_labels = []
    row_groups = []
    llm_results = []  # list of (results, models), one entry per LLM, in row order
    hlines_after_rows = set()
    for group in llm_groups:
        start = len(row_labels)
        for llm in group["llms"]:
            row_labels.append(llm_to_text.get(llm, llm))
            llm_results.append(read_llm_results(llm, group))
        row_groups.append(list(range(start, len(row_labels))))
        hlines_after_rows.add(len(row_labels) - 1)
    hlines_after_rows.discard(len(row_labels) - 1)  # no trailing hline after the last group

    for method in methods_to_consider:
        row_data, row_stds = [], []
        for group_indices in row_groups:
            family_llm_results = [llm_results[i] for i in group_indices]
            family_means, family_stds = average_family_ranks(family_llm_results, method)
            row_data.extend(family_means)
            row_stds.extend(family_stds)
        data = np.array(row_data)
        stds = np.array(row_stds)

        out_file = f"{output_path}/{safe_filename_label(method_names.get(method, method))}_ranks.txt"
        metrics_to_latex(out_file, data, row_labels, stds=stds, row_groups=row_groups, hlines_after_rows=hlines_after_rows)
        print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
