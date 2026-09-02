import os
from collections import defaultdict

import numpy as np

# This script is tailored to the settings used in run_reduced_tags.sh.
EXPERIMENTAL_SERIES = "main_evaluation_reduced_tags"
IDENTIFIERS = [
    "bnlearn_cancer",
    "bnlearn_earthquake",
    "bnlearn_survey",
    "bnlearn_asia",
    "lucas",
    "bnlearn_child",
    "bnlearn_alarm",
    "bnlearn_insurance",
    "bnlearn_hailfinder",
    "bnlearn_hepar2",
    "bnlearn_win95pts",
]
SEEDS = list(range(10))
FEWER_TAGS = [0.00, 0.25, 0.5, 0.75]

# run_reduced_tags.sh configuration
LLM = "anthropic--claude-opus-4.6"
ORDER_DATA = "random"
NR_SAMPLES = 10000
PC_INDEP_TEST = "chisq"
PC_ALPHA = 0.05
GES_SCORE_FUNC = "local_score_BDeu"
MIN_SAMPLES = 2
COMPUTE_MIN_PROB_THRESHOLD = True
MIN_PROB_THRESHOLD_DEFAULT = 0.5
ANTI_TAGS = False
REMOVE_DUPLICATES = True
REMOVE_SINGULAR_TAGS = False
PRIOR_ON_WEIGHT = False
ALWAYS_MEEKS = False
REDIRECT_EXISTING_EDGES = True
REDIRECTING_STRATEGY = 0
INCLUDE_CURRENT_EDGE_AS_EVIDENCE = True
INCLUDE_REDIRECTED_EDGES_IN_EDGE_COUNT = True
MIN_PROB_REDIRECTING = 0.6

# Which method row from eval.csv to report.
METHOD = "tag_pc_0_on_ges"
METRIC_NAMES = [
    "SHD",
    "SHD_double",
    "SID_min",
    "SID_max",
    "Precision",
    "Recall",
    "F1",
]
METRIC_OPTIMIZATION = ["min", "min", "min", "min", "max", "max", "max"]
SID_NA_DATASETS = {"bnlearn_hepar2", "bnlearn_win95pts", "bnlearn_hailfinder"}

DATASET_NAMES = {
    "bnlearn_child": "Child",
    "bnlearn_earthquake": "Earthquake",
    "bnlearn_insurance": "Insurance",
    "bnlearn_survey": "Survey",
    "bnlearn_asia": "Asia",
    "bnlearn_cancer": "Cancer",
    "bnlearn_alarm": "Alarm",
    "lucas": "Lucas",
    "bnlearn_hepar2": "Hepar2",
    "bnlearn_win95pts": "Win95Pts",
    "bnlearn_hailfinder": "Hailfinder",
}


def read_eval_csv(file_path):
    with open(file_path, "r", encoding="utf-8") as handle:
        rows = [line.strip().split(",") for line in handle if line.strip()]
    methods = [row[0] for row in rows]
    metrics = np.array([row[1:] for row in rows], dtype=float)
    return methods, metrics


def build_eval_path(identifier, seed, fewer_tags):
    fewer_tags_str = str(int(fewer_tags * 100)).zfill(2)
    path = (
        f"results/{EXPERIMENTAL_SERIES}/{identifier}/{ORDER_DATA}/{seed}/"
        f"{PC_INDEP_TEST}_{PC_ALPHA}_{GES_SCORE_FUNC}_{NR_SAMPLES}/{LLM}/"
        f"{ANTI_TAGS}_{REMOVE_DUPLICATES}_{REMOVE_SINGULAR_TAGS}_{PRIOR_ON_WEIGHT}_"
        f"{MIN_SAMPLES}_{COMPUTE_MIN_PROB_THRESHOLD}_{MIN_PROB_THRESHOLD_DEFAULT}_{fewer_tags_str}/"
        f"{ALWAYS_MEEKS}_{REDIRECT_EXISTING_EDGES}_{REDIRECTING_STRATEGY}_"
        f"{INCLUDE_CURRENT_EDGE_AS_EVIDENCE}_{INCLUDE_REDIRECTED_EDGES_IN_EDGE_COUNT}_"
        f"{MIN_PROB_REDIRECTING}/_tagging_alg0/eval.csv"
    )
    return path


def collect_metric_scores():
    metric_scores = defaultdict(lambda: defaultdict(list))
    missing = []

    for fewer_tags in FEWER_TAGS:
        for identifier in IDENTIFIERS:
            for seed in SEEDS:
                eval_path = build_eval_path(identifier, seed, fewer_tags)
                if not os.path.exists(eval_path):
                    missing.append(eval_path)
                    continue

                methods, metrics = read_eval_csv(eval_path)
                if METHOD not in methods:
                    raise ValueError(f"Method {METHOD} not found in {eval_path}")

                method_idx = methods.index(METHOD)
                metric_scores[fewer_tags][identifier].append(metrics[method_idx])

    return metric_scores, missing


def print_average_f1_by_config(metric_scores):
    print("Average F1 by fewer-tags config")
    print(f"Method: {METHOD}")
    print("-" * 42)
    for fewer_tags in FEWER_TAGS:
        all_values = []
        for identifier in IDENTIFIERS:
            all_values.extend([values[-1] for values in metric_scores[fewer_tags][identifier]])
        avg = np.mean(all_values) if len(all_values) > 0 else np.nan
        print(f"fewer_tags={fewer_tags:.2f} -> avg_f1={avg:.4f}")
    print()


def print_dataset_table(metric_scores):
    print("Per-dataset metrics table (4 fewer-tags configs per dataset)")
    print(f"Method: {METHOD}")
    print("-" * 205)

    header = (
        f"{'Dataset':<14} | {'fewer_tags':<10} | "
        f"{'SHD':<20} | {'SHD_double':<20} | {'SID_min':<20} | {'SID_max':<20} | "
        f"{'Precision':<20} | {'Recall':<20} | {'F1':<20}"
    )
    print(header)
    print("-" * len(header))

    for identifier in IDENTIFIERS:
        dataset_label = DATASET_NAMES.get(identifier, identifier)
        for idx, fewer_tags in enumerate(FEWER_TAGS):
            vals = metric_scores[fewer_tags][identifier]
            if len(vals) > 0:
                vals_array = np.array(vals)
                avg_metrics = np.nanmean(vals_array, axis=0)
                std_metrics = np.nanstd(vals_array, axis=0)
            else:
                avg_metrics = np.array([np.nan] * len(METRIC_NAMES))
                std_metrics = np.array([np.nan] * len(METRIC_NAMES))
            if identifier in SID_NA_DATASETS:
                avg_metrics[2:4] = np.nan
                std_metrics[2:4] = np.nan
            name_col = dataset_label if idx == 0 else ""
            metric_strs = [
                (
                    f"{avg_metrics[i]:.4f} +/- {std_metrics[i]:.4f}"
                    if not np.isnan(avg_metrics[i])
                    else "NA"
                )
                for i in range(len(METRIC_NAMES))
            ]
            print(
                f"{name_col:<14} | {fewer_tags:<10.2f} | "
                f"{metric_strs[0]:<20} | {metric_strs[1]:<20} | {metric_strs[2]:<20} | {metric_strs[3]:<20} | "
                f"{metric_strs[4]:<20} | {metric_strs[5]:<20} | {metric_strs[6]:<20}"
            )
        print("-" * len(header))


def write_dataset_table_latex(metric_scores, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{l|c|ccccccc}\n")
        handle.write(
            " & Fewer Tags & SHD & SHD\\textsubscript{double} & SID\\textsubscript{min} & SID\\textsubscript{max} & Precision & Recall & F\\textsubscript{1} \\\\\n"
        )

        for identifier in IDENTIFIERS:
            dataset_label = DATASET_NAMES.get(identifier, identifier)
            handle.write("\\hline \\hline\n")
            handle.write(f"{dataset_label} & & & & & & & & \\\\\n")
            handle.write("\\hline\n")

            dataset_means = []
            dataset_stds = []
            counts = []
            for idx, fewer_tags in enumerate(FEWER_TAGS):
                vals = metric_scores[fewer_tags][identifier]
                if len(vals) > 0:
                    vals_array = np.array(vals)
                    avg_metrics = np.nanmean(vals_array, axis=0)
                    std_metrics = np.nanstd(vals_array, axis=0)
                else:
                    avg_metrics = np.array([np.nan] * len(METRIC_NAMES))
                    std_metrics = np.array([np.nan] * len(METRIC_NAMES))
                if identifier in SID_NA_DATASETS:
                    avg_metrics[2:4] = np.nan
                    std_metrics[2:4] = np.nan

                dataset_means.append(avg_metrics)
                dataset_stds.append(std_metrics)
                counts.append(len(vals))

            dataset_means = np.array(dataset_means)
            dataset_stds = np.array(dataset_stds)

            for idx, fewer_tags in enumerate(FEWER_TAGS):
                metric_strs = []
                for metric_idx in range(len(METRIC_NAMES)):
                    mean_val = dataset_means[idx, metric_idx]
                    std_val = dataset_stds[idx, metric_idx]
                    if np.isnan(mean_val):
                        metric_strs.append("-")
                        continue
                    metric_str = "$"
                    metric_str += f"{mean_val:.2f}"
                    metric_str += f" {{\\scriptstyle \\pm {std_val:.2f}}}$"
                    metric_strs.append(metric_str)

                row = (
                    f" & {fewer_tags:.2f} & "
                    f"{metric_strs[0]} & {metric_strs[1]} & {metric_strs[2]} & {metric_strs[3]} & "
                    f"{metric_strs[4]} & {metric_strs[5]} & {metric_strs[6]} \\\\"
                )
                handle.write(row + "\n")

        handle.write("\\end{tabular}\n")

    print(f"Saved LaTeX table to {output_path}")


def main():
    metric_scores, missing_paths = collect_metric_scores()

    if missing_paths:
        print(f"Warning: missing {len(missing_paths)} eval files.")
        print("Showing first 5 missing paths:")
        for path in missing_paths[:5]:
            print(f"  {path}")
        print()

    print_average_f1_by_config(metric_scores)
    print_dataset_table(metric_scores)

    latex_path = f"results/{EXPERIMENTAL_SERIES}/reduced_tags_dataset_metrics.tex"
    write_dataset_table_latex(metric_scores, latex_path)


if __name__ == "__main__":
    main()
