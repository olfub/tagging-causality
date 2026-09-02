import json
import os

import numpy as np
from scipy.stats import rankdata

# must match run_eval.sh / results_eval.py (main_evaluation) and E7_llm_prompt_versions.sh
identifiers = ["bnlearn_child", "bnlearn_earthquake", "bnlearn_insurance", "bnlearn_survey", "bnlearn_asia",
               "bnlearn_cancer", "bnlearn_alarm", "lucas", "bnlearn_hepar2", "bnlearn_win95pts", "bnlearn_hailfinder"]
seeds = list(range(10))

# these 6 stay fixed at their own main_evaluation best config (same as results_eval.py table 0)
fixed_methods = ["pc", "ges", "typed_pc_naive", "typed_pc_maj", "tag_pc_1", "tag_pc_0"]
# this is the one method that gets a row per prompt variant instead of a single row
varied_method = "tag_pc_0_on_ges"

method_names = {
    "pc": "PC",
    "ges": "GES",
    "typed_pc_naive": "Typed-PC (Naive)",
    "typed_pc_maj": "Typed-PC (Maj.)",
    "tag_pc_1": "Tagged-PC (AntiV)",
    "tag_pc_0": "Tagged-PC",
}
varied_method_label = "Tagged-GES"

base_llm = "anthropic--claude-opus-4.6"

# same prompt-variation categories as E7_llm_prompt_versions.py
variation_groups = [
    {
        "label": "Prompt Decomposition",
        "variants": {
            "prompt_decomposition_tag_definition": "Tag Definition",
            "prompt_decomposition_specificityTradeoff": "Specificity Tradeoff",
            "prompt_decomposition_downstreamPurpose": "Downstream Purpose",
            "prompt_decomposition_discriminativeness": "Discriminativeness",
            "prompt_decomposition_multipleTags": "Multiple Tags",
            "prompt_decomposition_tag_balance": "Tag Balance",
            "prompt_decomposition_noDuplicated": "No Duplicated",
        },
    },
    {
        "label": "Paraphrase (Register)",
        "variants": {
            "paraphrase_register_bullets": "Bullets",
            "paraphrase_register_terse": "Terse",
            "paraphrase_register_verbose": "Verbose",
        },
    },
    {
        "label": "Paraphrase (Politeness)",
        "variants": {
            "paraphrase_politeness_polite": "Polite",
            "paraphrase_politeness_demanding": "Demanding",
            "paraphrase_politeness_imperative": "Imperative",
        },
    },
    {
        "label": "Tag Budget per Variable",
        "variants": {
            "budget_tags_per_var_2": "Budget = 2",
            "budget_tags_per_var_3": "Budget = 3",
            "budget_tags_per_var_4": "Budget = 4",
            "budget_tags_per_var_6": "Budget = 6",
            "budget_tags_per_var_8": "Budget = 8",
            "budget_tags_per_var_10": "Budget = 10",
        },
    },
]

# the exact config E7_llm_prompt_versions.sh uses for every variant (identical to
# results/main_evaluation/_configs/_best_config_tag_pc_0_on_ges.json, only "llm" changes)
e7_config_template = {
    "order_data": "random",
    "nr_samples": 10000,
    "pc_indep_test": "chisq",
    "pc_alpha": 0.05,
    "ges_indep_test": "local_score_BDeu",
    "min_samples": 2,
    "compute_min_prob_threshold": True,
    "min_prob_threshold_default": 0.5,
    "anti_tags": False,
    "remove_duplicates": True,
    "remove_singular_tags": False,
    "prior_on_weight": False,
    "always_meeks": False,
    "redirect_existing_edges": True,
    "redirecting_strategy": 0,
    "min_prob_redirecting": 0.6,
    "include_current_edge_as_evidence": True,
    "include_redirected_edges_in_edge_count": True,
}


def read_eval_csv(file):
    with open(file, "r") as f:
        metrics, models = [], []
        for line in f.read().splitlines():
            parts = line.split(",")
            models.append(parts[0])
            metrics.append(parts[1:])
        return np.array(metrics, dtype=float), models


def eval_csv_path(experimental_series, config, identifier, seed):
    fewer_tags_str = "00"
    path = f"results/{experimental_series}/{identifier}/{config['order_data']}/{seed}/{config['pc_indep_test']}_{config['pc_alpha']}_{config['ges_indep_test']}_{config['nr_samples']}"
    path_llm = f"{path}/{config['llm']}"
    path_tagging = (
        f"{path_llm}/{config['anti_tags']}_{config['remove_duplicates']}_{config['remove_singular_tags']}_{config['prior_on_weight']}_"
        f"{config['min_samples']}_{config['compute_min_prob_threshold']}_{config['min_prob_threshold_default']}_{fewer_tags_str}"
    )
    path_tagging_alg0 = (
        f"{path_tagging}/{config['always_meeks']}_{config['redirect_existing_edges']}_{config['redirecting_strategy']}_"
        f"{config['include_current_edge_as_evidence']}_{config['include_redirected_edges_in_edge_count']}_{config['min_prob_redirecting']}"
    )
    return f"{path_tagging_alg0}/_tagging_alg0/eval.csv"


def read_method_values(experimental_series, config, method_name):
    # returns dict[(identifier, seed)] -> length-7 metric vector for this one method
    values = {}
    for identifier in identifiers:
        for seed in seeds:
            eval_file = eval_csv_path(experimental_series, config, identifier, seed)
            eval_result, models = read_eval_csv(eval_file)
            vec = eval_result[models.index(method_name)].copy()
            if identifier in ["bnlearn_hepar2", "bnlearn_win95pts", "bnlearn_hailfinder"]:
                # SID is not computed for these (too large), stored as 0 -> mask out
                assert vec[2] == 0 and vec[3] == 0
                vec[2] = np.nan
                vec[3] = np.nan
            values[(identifier, seed)] = vec
    return values


def metrics_to_ranks(metrics):
    # metrics: (n_methods, 7); first 4 columns minimized (SHD/SID), last 3 maximized (P/R/F1)
    ranks = np.zeros_like(metrics)
    for i in range(metrics.shape[1]):
        if i < 4:
            ranks[:, i] = rankdata(metrics[:, i], method="min")
        else:
            ranks[:, i] = rankdata(-metrics[:, i], method="min")
    return ranks


def average_ranks_for_variant(fixed_values, variant_values):
    # fixed_values: dict[method] -> dict[(ds, seed)] -> vec, for the 6 fixed methods
    # variant_values: dict[(ds, seed)] -> vec, for this one variant's tag_pc_0_on_ges
    # returns two (7, 7) matrices (mean and std of the ranks): rows = fixed_methods + [varied_method], cols = metrics
    # ranks are averaged over the datasets within one seed, then aggregated over the seeds
    # (same convention as results_eval.py: the std is the spread of the per-seed average ranks)
    per_seed_average_ranks = []
    for seed in seeds:
        per_ds_ranks = []
        for identifier in identifiers:
            rows = [fixed_values[m][(identifier, seed)] for m in fixed_methods]
            rows.append(variant_values[(identifier, seed)])
            per_ds_ranks.append(metrics_to_ranks(np.array(rows)))
        per_seed_average_ranks.append(np.nanmean(np.array(per_ds_ranks), axis=0))
    per_seed_average_ranks = np.array(per_seed_average_ranks)
    return np.nanmean(per_seed_average_ranks, axis=0), np.nanstd(per_seed_average_ranks, axis=0)


def metrics_to_latex(file_path, data, row_labels, stds=None, hlines_after_rows=None):
    # data: (n_rows, 7) of average ranks; lower rank is always better, bold the min per column
    bf_values = np.min(data, axis=0)
    with open(file_path, "w") as f:
        f.write("\\begin{tabular}{l|ccccccc}\n")
        f.write("Method & SHD & SHD\\textsubscript{double} & SID\\textsubscript{min} & SID\\textsubscript{max} & Precision & Recall & F\\textsubscript{1} \\\\\n")
        f.write(" & Ranks & Ranks & Ranks & Ranks & Ranks & Ranks & Ranks \\\\\n")
        f.write("\\hline\n")
        for i, label in enumerate(row_labels):
            f.write(f"{label} & ")
            for j in range(data.shape[1]):
                value = data[i, j]
                cell = f"\\mathbf{{{value:.2f}}}" if value == bf_values[j] else f"{value:.2f}"
                if stds is not None:
                    cell += f" {{\\scriptstyle \\pm {stds[i, j]:.2f}}}"
                sep = " \\\\\n" if j == data.shape[1] - 1 else " & "
                f.write(f"${cell}$" + sep)
            if hlines_after_rows is not None and i in hlines_after_rows:
                f.write("\\hline\n")
        f.write("\\end{tabular}\n")


def main():
    output_path = "results/E7_llm_prompt_versions/_result_tables"
    os.makedirs(output_path, exist_ok=True)

    # read the 6 fixed methods, each at its own main_evaluation best config
    fixed_values = {}
    for method in fixed_methods:
        with open(f"results/main_evaluation/_configs/_best_config_{method}.json", "r") as f:
            config = json.load(f)
        fixed_values[method] = read_method_values("main_evaluation", config, method)

    # read tag_pc_0_on_ges for the baseline (default prompt) and every prompt variant
    variant_llms = [(None, "Baseline (Default Prompt)")]
    for group in variation_groups:
        for variant, display_name in group["variants"].items():
            variant_llms.append((variant, display_name))

    variant_values = {}
    for variant, display_name in variant_llms:
        if variant is None:
            # the baseline (default prompt) already exists as main_evaluation's own
            # tag_pc_0_on_ges best config -- read it from there directly instead of the
            # separately re-run E7 copy, since two independent runs (even same seed/config)
            # are not bit-identical, which would otherwise break the "same ranks as
            # before" invariant for this row
            with open("results/main_evaluation/_configs/_best_config_tag_pc_0_on_ges.json", "r") as f:
                config = json.load(f)
            variant_values[display_name] = read_method_values("main_evaluation", config, varied_method)
        else:
            config = dict(e7_config_template)
            config["llm"] = f"{base_llm}__{variant}"
            variant_values[display_name] = read_method_values("E7_llm_prompt_versions", config, varied_method)

    # one independent 7-way ranking per variant (variants are never ranked against each other)
    row_labels = [method_names[m] for m in fixed_methods]
    row_data = None
    row_stds = None
    variant_labels = []

    # hlines after the fixed-methods block, then after the baseline row, then after each variant group
    hlines_after_rows = {len(fixed_methods) - 1}
    group_sizes = [1] + [len(g["variants"]) for g in variation_groups]  # baseline is its own group of 1
    row_idx = len(fixed_methods) - 1
    for gsize in group_sizes:
        row_idx += gsize
        hlines_after_rows.add(row_idx)
    hlines_after_rows.discard(row_idx)  # no trailing hline after the very last row

    for variant, display_name in variant_llms:
        avg_ranks, std_ranks = average_ranks_for_variant(fixed_values, variant_values[display_name])
        if row_data is None:
            # first variant is the baseline: its ranks for the 6 fixed methods equal the
            # original main_evaluation table 0 exactly (same underlying data), so we can
            # use this single computation for both the fixed rows and the baseline row
            row_data = list(avg_ranks[:-1])
            row_stds = list(std_ranks[:-1])
        row_data.append(avg_ranks[-1])
        row_stds.append(std_ranks[-1])
        variant_labels.append(f"{varied_method_label} ({display_name})")

    row_labels += variant_labels
    data = np.array(row_data)
    stds = np.array(row_stds)

    out_file = f"{output_path}/all_ranks_per_prompt_variant.txt"
    metrics_to_latex(out_file, data, row_labels, stds=stds, hlines_after_rows=hlines_after_rows)
    print(f"Wrote {out_file}")
    print("(the 'Baseline (Default Prompt)' row should match Tagged-GES's row in "
          "results/main_evaluation/_result_tables/all_ranks.txt)")


if __name__ == "__main__":
    main()
