import ast
import csv
import json
import os

import networkx as nx
import numpy as np

import util
from discovery.pc_tagged import get_edges

datasets = [
    "bnlearn_cancer", "bnlearn_earthquake", "bnlearn_survey",
    "bnlearn_asia", "lucas", "bnlearn_child",
    "bnlearn_insurance", "bnlearn_alarm", "bnlearn_hailfinder", "bnlearn_hepar2", "bnlearn_win95pts"
]

dataset_names = {"bnlearn_child": "Child", "bnlearn_earthquake": "Earthquake", "bnlearn_insurance": "Insurance",
                 "bnlearn_survey": "Survey", "bnlearn_asia": "Asia", "bnlearn_cancer": "Cancer",
                 "bnlearn_alarm": "Alarm", "lucas": "Lucas", "bnlearn_hepar2": "Hepar2", "bnlearn_win95pts": "Win95Pts", "bnlearn_hailfinder": "Hailfinder"}

experimental_series = "main_evaluation"
seeds = list(range(10))
fewer_tags_str = "00"

config = json.load(open(f"results/{experimental_series}/_configs/_best_config_tag_pc_0_on_ges.json", "r"))
order_data = config["order_data"]
llm = config["llm"]
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


def tagging_alg0_folder(identifier, seed):
    # mirrors the path construction in run.py for the tag_pc_0_on_ges result
    path = f"results/{experimental_series}/{identifier}/{order_data}/{seed}/{pc_indep_test}_{pc_alpha}_{ges_indep_test}_{nr_samples}"
    path_llm = f"{path}/{llm}"
    path_tagging = f"{path_llm}/{anti_tags}_{remove_duplicates}_{remove_singular_tags}_{prior_on_weight}_{min_samples}_{compute_min_prob_threshold}_{min_prob_threshold_default}_{fewer_tags_str}"
    path_tagging_alg0 = f"{path_tagging}/{always_meeks}_{redirect_existing_edges}_{redirecting_strategy}_{include_current_edge_as_evidence}_{include_redirected_edges_in_edge_count}_{min_prob_redirecting}"
    return f"{path_tagging_alg0}/_tagging_alg0"


def load_raw_rows(csv_path):
    # each row is written by util.save_info as:
    # [pred, [var_a, var_b], score, direction, direction_valid, (forward_tag_pairs, backward_tag_pairs), from_redirect_pass]
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            _pred, edge_vars, score, direction, _direction_valid, extra, from_redirect_pass = row
            forward_evidence, backward_evidence = ast.literal_eval(extra)
            rows.append({
                "edge": tuple(ast.literal_eval(edge_vars)),
                "score": float(score),
                "direction_forward": ast.literal_eval(direction),
                "forward_evidence": forward_evidence,
                "backward_evidence": backward_evidence,
                "from_redirect_pass": ast.literal_eval(from_redirect_pass),
            })
    return rows


def load_named_graphs(folder):
    adjacency_matrices = list(np.load(f"{folder}/adjacency_matrices.npy"))
    with open(f"{folder}/names.txt", "r") as f:
        names = [line.strip() for line in f]
    return {name: adj for name, adj in zip(names, adjacency_matrices)}


def ges_gt_skeleton(ges_adj):
    # reconstructs the same gt_skeleton run.py builds for tag_pc_0_on_ges (run.py:333)
    graph_ges = nx.DiGraph()
    graph_ges.add_nodes_from(range(ges_adj.shape[0]))
    graph_ges.add_edges_from(
        (i, j) for i in range(ges_adj.shape[0]) for j in range(ges_adj.shape[1]) if ges_adj[i, j] != 0
    )
    return util.skeleton_to_cg_graph(graph_ges, undirected=False)


def ges_undirected_pairs(ges_adj):
    # distinct (i, j) with i < j for every edge still undirected in GES's output,
    # i.e. available to tagging
    _, undirected_edges = get_edges(ges_gt_skeleton(ges_adj).G.graph)
    return [(i, j) for (i, j) in undirected_edges if i < j]  # get_edges lists both (i,j) and (j,i)


def collect_tag_pc_0_on_ges_info(seeds):
    # collect all information on how tagging directed edges that were undirected in
    # GES's output, across all datasets and seeds
    info = {}
    edge_status = []
    for seed in seeds:
        info[seed] = {}
        for identifier in datasets:
            folder = tagging_alg0_folder(identifier, seed)
            csv_path = f"{folder}/tag_pc_0_on_ges_info.csv"
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Missing info file for {identifier} seed {seed}: {csv_path}")
            rows = load_raw_rows(csv_path)

            variables, var_labels, _tags, edges, _positions, _samples = util.load_data(
                identifier, nr_samples, order_data=order_data, seed=seed
            )
            true_graph = util.create_model(variables, edges, seed).graph
            name_to_idx = {name: i for i, name in enumerate(var_labels)}
            named_graphs = load_named_graphs(folder)

            records = []
            for row in rows:
                var_a, var_b = row["edge"]
                source, target = (var_a, var_b) if row["direction_forward"] else (var_b, var_a)
                source_idx, target_idx = name_to_idx[source], name_to_idx[target]

                # spurious means that the edge is not present at all in the true graph,
                # i.e., the skeleton was already wrong
                if true_graph.has_edge(source_idx, target_idx):
                    label = "correct"
                elif true_graph.has_edge(target_idx, source_idx):
                    label = "reversed"
                else:
                    label = "spurious"

                if row["direction_forward"]:
                    supporting_evidence, opposing_evidence = row["forward_evidence"], row["backward_evidence"]
                else:
                    supporting_evidence, opposing_evidence = row["backward_evidence"], row["forward_evidence"]

                records.append({
                    "seed": seed,
                    "dataset": identifier,
                    "source": source,
                    "target": target,
                    "score": row["score"],
                    "label": label,
                    "correct": label == "correct",
                    "from_redirect_pass": row["from_redirect_pass"],
                    "supporting_evidence": supporting_evidence,
                    "opposing_evidence": opposing_evidence,
                })
            info[seed][identifier] = records

            # ignore redirect decisions
            resolved_by_pair = {
                frozenset((r["source"], r["target"])): r
                for r in records if not r["from_redirect_pass"]
            }
            for i, j in ges_undirected_pairs(named_graphs["ges"]):
                var_i, var_j = var_labels[i], var_labels[j]
                is_spurious = not (true_graph.has_edge(i, j) or true_graph.has_edge(j, i))
                resolved_rec = resolved_by_pair.get(frozenset((var_i, var_j)))
                if resolved_rec is None:
                    edge_status.append({
                        "seed": seed, "dataset": identifier,
                        "var_a": var_i, "var_b": var_j, "directed": False,
                        "category": "spurious" if is_spurious else "undirected",
                    })
                else:
                    category = "spurious" if is_spurious else ("correct" if resolved_rec["label"] == "correct" else "incorrect")
                    edge_status.append({
                        "seed": seed, "dataset": identifier,
                        "var_a": resolved_rec["source"], "var_b": resolved_rec["target"], "directed": True,
                        "category": category,
                    })
    return info, edge_status


def flatten(info):
    return [r for seed_dict in info.values() for records in seed_dict.values() for r in records]


def top_supporting_tag_pair(evidence):
    if not evidence:
        return None
    tag1, tag2, _count = max(evidence, key=lambda e: e[2])
    return tag1, tag2


def _latex_escape(text):
    return text.replace("_", "\\_")


def format_evidence_latex(evidence):
    if not evidence:
        return "none"
    parts = [f"{_latex_escape(tag1)} $\\rightarrow$ {_latex_escape(tag2)} (n={count:.3g})" for tag1, tag2, count in evidence]
    return ", ".join(parts)


def compact_summary_latex(aggregated):
    # print information on incorrect predictions using tagging
    lines = []
    current_dataset = None
    for a in sorted(aggregated, key=lambda a: (dataset_names.get(a["dataset"], a["dataset"]), -a["count"])):
        dataset_label = dataset_names.get(a["dataset"], a["dataset"])
        if dataset_label != current_dataset:
            lines.append(f" & \\textbf{{{dataset_label}}} & & \\nonumber \\\\")
            current_dataset = dataset_label
        source = _latex_escape(a["source"])
        target = _latex_escape(a["target"])
        pair = top_supporting_tag_pair(a["supporting_evidence"])
        if pair is None:
            tag_str = "none"
        else:
            tag1, tag2 = _latex_escape(pair[0]), _latex_escape(pair[1])
            tag_str = f"\\text{{``{tag1}''}}~\\rightarrow~\\text{{``{tag2}''}}"
        lines.append(f"\\text{{``{source}''}} &~\\rightarrow~ \\text{{``{target}''}} && ({tag_str}) \\nonumber \\\\")
    return "\n".join(lines)


def detailed_summary_latex(aggregated, seeds):
    # like compact_summary_latex, but with full comprehensive information
    lines = []
    current_dataset = None
    for a in sorted(aggregated, key=lambda a: (dataset_names.get(a["dataset"], a["dataset"]), -a["count"])):
        dataset_label = dataset_names.get(a["dataset"], a["dataset"])
        if dataset_label != current_dataset:
            lines.append(f" & \\textbf{{{dataset_label}}} & & \\nonumber \\\\")
            current_dataset = dataset_label
        source = _latex_escape(a["source"])
        target = _latex_escape(a["target"])
        seeds_str = ",".join(str(s) for s in a["seeds"])
        lines.append(
            f"\\text{{``{source}''}} &~\\rightarrow~ \\text{{``{target}''}} && "
            f"({a['label']}, {a['count']}/{len(seeds)} \\text{{ seeds: }} {seeds_str}, "
            f"\\text{{avg score}}={a['avg_score']:.4f}) \\nonumber \\\\"
        )
        lines.append(f"& \\text{{supporting: }} {format_evidence_latex(a['supporting_evidence'])} \\nonumber \\\\")
        lines.append(f"& \\text{{opposing: }} {format_evidence_latex(a['opposing_evidence'])} \\nonumber \\\\")
    return "\n".join(lines)


def _format_status_counts_cell(counts):
    return (
        f"\\textcolor{{blue}}{{{counts['correct']}}}/"
        f"\\textcolor{{red}}{{{counts['incorrect']}}}/"
        f"\\textcolor{{black}}{{{counts['undirected']}}}"
    )


def _format_status_avg_cell(counts, n):
    return (
        f"\\textcolor{{blue}}{{{counts['correct'] / n:.1f}}}/"
        f"\\textcolor{{red}}{{{counts['incorrect'] / n:.1f}}}/"
        f"\\textcolor{{black}}{{{counts['undirected'] / n:.1f}}}"
    )


def spurious_cases_comment(edge_status):
    # spurious cases (no true edge in either direction) are rare enough that they don't
    # earn their own table column; document them here instead of in the table
    spurious = [e for e in edge_status if e["category"] == "spurious"]
    lines = [f"% {len(spurious)} spurious case(s) (no true edge in either direction), excluded from the table above:"]
    for e in spurious:
        dataset_label = dataset_names.get(e["dataset"], e["dataset"])
        connector = "->" if e["directed"] else "--"
        lines.append(f"%   [{dataset_label}, seed {e['seed']}] {e['var_a']} {connector} {e['var_b']}")
    return "\n".join(lines)


def edge_status_counts_table_latex(edge_status, seeds):
    """
    Datasets on rows, seeds on columns. Each cell is a/b/c: a = correctly directed
    (blue), b = incorrectly directed (red), c = left undirected (black) -- counted over
    every edge that was still undirected in GES's CPDAG before tagging, based on what
    tagging then did with it. Spurious cases (no true edge in either direction) are rare
    and are documented in a comment below the table instead of taking up a column.
    """
    empty_counts = {"correct": 0, "incorrect": 0, "undirected": 0, "spurious": 0}
    counts = {}
    for e in edge_status:
        key = (e["seed"], e["dataset"])
        counts.setdefault(key, dict(empty_counts))
        counts[key][e["category"]] += 1

    header = " & ".join(str(seed) for seed in seeds)
    lines = [
        "\\begin{tabular}{l" + "c" * len(seeds) + "|c}",
        "    \\toprule",
        f"    Dataset & {header} & Average \\\\",
        "    \\midrule",
    ]
    for d in datasets:
        row_cells = []
        total = dict(empty_counts)
        for seed in seeds:
            c = counts.get((seed, d), empty_counts)
            row_cells.append(_format_status_counts_cell(c))
            for k in total:
                total[k] += c[k]
        row = " & ".join(row_cells)
        lines.append(f"    {dataset_names.get(d, d)} & {row} & {_format_status_avg_cell(total, len(seeds))} \\\\")
    lines.append("    \\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    lines.append(spurious_cases_comment(edge_status))
    return "\n".join(lines)


def aggregate_incorrect(records):
    # group repeated occurrences of the same wrongly-directed edge
    groups = {}
    for r in records:
        key = (r["dataset"], r["source"], r["target"])
        groups.setdefault(key, []).append(r)

    aggregated = []
    for (dataset, source, target), recs in groups.items():
        aggregated.append({
            "dataset": dataset,
            "source": source,
            "target": target,
            "label": recs[0]["label"],
            "count": len(recs),
            "seeds": sorted(r["seed"] for r in recs),
            "avg_score": sum(r["score"] for r in recs) / len(recs),
            "supporting_evidence": recs[0]["supporting_evidence"],
            "opposing_evidence": recs[0]["opposing_evidence"],
        })
    aggregated.sort(key=lambda a: (a["count"], a["avg_score"]), reverse=True)
    return aggregated


if __name__ == "__main__":
    info, edge_status = collect_tag_pc_0_on_ges_info(seeds)

    table_dir = "E_results/E_4_bad_directs"
    os.makedirs(table_dir, exist_ok=True)

    # this analysis is not targeted at the redirection pass
    non_redirect_records = [r for r in flatten(info) if not r["from_redirect_pass"]]
    incorrect_records = [r for r in non_redirect_records if not r["correct"]]
    aggregated = aggregate_incorrect(incorrect_records)

    lines = [
        "% compact summary: dataset, wrong edge, top supporting tag",
        compact_summary_latex(aggregated),
        "",
        "% detailed summary: label, seeds it occurred in, avg score, full evidence",
        detailed_summary_latex(aggregated, seeds),
    ]
    with open(f"{table_dir}/wrong_directions.tex", "w") as f:
        f.write("\n".join(lines) + "\n")

    with open(f"{table_dir}/edge_status_counts.tex", "w") as f:
        f.write(edge_status_counts_table_latex(edge_status, seeds) + "\n")
