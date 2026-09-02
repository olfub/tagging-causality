import os

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTAL_SERIES = "E3_different_thresholds"

IDENTIFIERS = [
    "bnlearn_child", "bnlearn_earthquake", "bnlearn_insurance", "bnlearn_survey",
    "bnlearn_asia", "bnlearn_cancer", "bnlearn_alarm", "lucas",
    "bnlearn_hepar2", "bnlearn_win95pts", "bnlearn_hailfinder",
]
LLM = "google--gemini-3-pro-preview"
ORDER_DATA = "random"
NR_SAMPLES = 10000
SEEDS = list(range(10))
PC_INDEP_TEST = "chisq"
PC_ALPHA = 0.05
GES_SCORE_FUNC = "local_score_BDeu"
MIN_SAMPLES = 1
COMPUTE_MIN_PROB_THRESHOLD = False
ANTI_TAGS = False
REMOVE_DUPLICATES = True
REMOVE_SINGULAR_TAGS = False
PRIOR_ON_WEIGHT = True
ALWAYS_MEEKS = True
REDIRECT_EXISTING_EDGES = False
# these are the redirect parameter placeholder which have to be used for loading
REDIRECTING_STRATEGY = 1
INCLUDE_CURRENT_EDGE_AS_EVIDENCE = False
INCLUDE_REDIRECTED_EDGES_IN_EDGE_COUNT = True
MIN_PROB_REDIRECTING = 0.6

FEWER_TAGS = 0.00
DIRECT_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

METHOD = "tag_pc_0_on_skel_v"
BASE_SKELETON = {
    "tag_pc_0_on_skel_v": "skel_v_meeks",
}[METHOD]

CUSTOM_PALETTE = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
                  '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC', '#493267']
COUNT_COLORS = {"correct": CUSTOM_PALETTE[4], "incorrect": CUSTOM_PALETTE[2], "abstained": CUSTOM_PALETTE[9]}


def result_folder(identifier, seed, fewer_tag, direct_threshold):
    fewer_tags_str = str(int(round(fewer_tag * 100))).zfill(2)
    base = f"results/{EXPERIMENTAL_SERIES}/{identifier}/{ORDER_DATA}/{seed}/{PC_INDEP_TEST}_{PC_ALPHA}_{GES_SCORE_FUNC}_{NR_SAMPLES}"
    path_llm = f"{base}/{LLM}"
    path_tagging = (
        f"{path_llm}/{ANTI_TAGS}_{REMOVE_DUPLICATES}_{REMOVE_SINGULAR_TAGS}_{PRIOR_ON_WEIGHT}_"
        f"{MIN_SAMPLES}_{COMPUTE_MIN_PROB_THRESHOLD}_{direct_threshold}_{fewer_tags_str}"
    )
    path_tagging_alg0 = (
        f"{path_tagging}/{ALWAYS_MEEKS}_{REDIRECT_EXISTING_EDGES}_{REDIRECTING_STRATEGY}_"
        f"{INCLUDE_CURRENT_EDGE_AS_EVIDENCE}_{INCLUDE_REDIRECTED_EDGES_IN_EDGE_COUNT}_{MIN_PROB_REDIRECTING}"
    )
    return f"{path_tagging_alg0}/_tagging_alg0"


def count_direction_decisions(folder, method_name, base_skeleton_name):
    # Compares pre tagging graph with post tagging graph
    # First: find all undirected edges in pre tagging
    # Load the ground truth graph as reference
    # Check whether undirected edges have been directed correctly, incorrectly, or left 
    # undirected (abstained)
    # This is designed for the true CPDAG analysis, so the skeleton should always be 
    # the same and there shoud always be a ground truth edge available as reference
    adj_path = f"{folder}/adjacency_matrices.npy"
    names_path = f"{folder}/names.txt"
    if not os.path.exists(adj_path) or not os.path.exists(names_path):
        raise FileNotFoundError(f"Missing adjacency_matrices.npy or names.txt in {folder}")

    adj = np.load(adj_path)
    with open(names_path, "r") as f:
        names = [line.strip() for line in f]
    index = {name: i for i, name in enumerate(names)}
    if base_skeleton_name not in index or method_name not in index or "true" not in index:
        raise ValueError(f"Missing required graph in {folder}")

    base_adj = adj[index[base_skeleton_name]]
    method_adj = adj[index[method_name]]
    true_adj = adj[index["true"]]

    n = base_adj.shape[0]
    correct = incorrect = abstained = 0
    for i in range(n):
        for j in range(i + 1, n):
            if base_adj[i, j] == 0 or base_adj[j, i] == 0:
                continue  # not an undirected edge, i.e., no tagging
            true_ij, true_ji = true_adj[i, j], true_adj[j, i]
            assert true_ij != 0 or true_ji != 0, (
                f"base-skeleton edge ({i}, {j}) in {folder} has no corresponding "
                "true edge in either direction; the base skeleton is derived from "
                "the true graph, so this should be impossible"
            )
            m_ij, m_ji = method_adj[i, j], method_adj[j, i]
            if m_ij != 0 and m_ji != 0:
                abstained += 1
                continue
            if m_ij != 0:
                chosen = true_ij
            elif m_ji != 0:
                chosen = true_ji
            else:
                raise ValueError(f"Edge disappeared; should never happen. Folder: {folder}, edge ({i}, {j})")
            if chosen != 0:
                correct += 1
            else:
                incorrect += 1
    return correct, incorrect, abstained


def main():
    path = f"E_results/E_3_different_thresholds"
    os.makedirs(path, exist_ok=True)

    # record the average over seeds for all direct thresholds
    records = {th: [] for th in DIRECT_THRESHOLDS}

    for direct_threshold in DIRECT_THRESHOLDS:
        for identifier in IDENTIFIERS:
            seed_runs = []
            for seed in SEEDS:
                folder = result_folder(identifier, seed, FEWER_TAGS, direct_threshold)
                counts = count_direction_decisions(folder, METHOD, BASE_SKELETON)
                correct, incorrect, abstained = counts
                seed_runs.append({
                    "correct": correct,
                    "incorrect": incorrect,
                    "abstained": abstained,
                })
            if not seed_runs:
                continue
            # average across seeds, but results should be identical per seed anyhow
            records[direct_threshold].append({
                "identifier": identifier,
                "n_seeds": len(seed_runs),
                "correct": np.mean([r["correct"] for r in seed_runs]),
                "incorrect": np.mean([r["incorrect"] for r in seed_runs]),
                "abstained": np.mean([r["abstained"] for r in seed_runs]),
            })

    # aggregate over datasets: sum of counts. Each entry in `runs` is already
    # averaged across that dataset's own seeds, so every dataset counts once
    # here regardless of seed count.
    summary = {}
    for direct_threshold in DIRECT_THRESHOLDS:
        runs = records[direct_threshold]
        if not runs:
            summary[direct_threshold] = None
            continue
        correct = sum(r["correct"] for r in runs)
        incorrect = sum(r["incorrect"] for r in runs)
        abstained = sum(r["abstained"] for r in runs)
        total = correct + incorrect + abstained

        summary[direct_threshold] = {
            "n_datasets": len(runs),
            "correct": correct,
            "incorrect": incorrect,
            "abstained": abstained,
            "correct_frac": correct / total if total > 0 else np.nan,
            "incorrect_frac": incorrect / total if total > 0 else np.nan,
            "abstained_frac": abstained / total if total > 0 else np.nan,
        }

    assert all(summary[th] is not None for th in DIRECT_THRESHOLDS), (
        "at least one direct_threshold has no data for any dataset; "
        "expected every threshold in DIRECT_THRESHOLDS to have been run"
    )
    thresholds_with_data = [th for th in DIRECT_THRESHOLDS if summary[th] is not None]

    # plot: correct / incorrect / abstained decision fractions vs threshold
    fig, ax = plt.subplots(figsize=(13, 8))
    bottoms = np.zeros(len(thresholds_with_data))
    for key in ["correct", "incorrect", "abstained"]:
        fracs = np.array([summary[th][f"{key}_frac"] for th in thresholds_with_data])
        ax.bar(
            thresholds_with_data, fracs, bottom=bottoms, width=0.04,
            color=COUNT_COLORS[key], label=key.capitalize(), edgecolor="white", linewidth=0.8, zorder=3,
        )
        bottoms += fracs
    ax.set_xlabel("Direct Threshold", fontsize=30)
    ax.set_ylabel("Fraction of Edges", fontsize=30)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.grid(True)
    ax.legend(fontsize=26, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    plt.tight_layout()
    counts_plot_file = f"{path}/threshold_counts.pdf"
    plt.savefig(counts_plot_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot to {counts_plot_file}")


if __name__ == "__main__":
    main()
