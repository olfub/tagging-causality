import os
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import util
from discovery.pc import pc
from discovery.pc_tagged import get_edges, get_edge_count


llm_to_text = {"openai--gpt-5.2": "GPT 5.2", "anthropic--claude-opus-4.6": "Claude 4.6", "google--gemini-3-pro-preview": "Gemini 3 Pro", "meta-llama--llama-3.3-70b-instruct": "Llama 3.3", "qwen--qwen3.5-397b-a17b": "Qwen 3.5", "z-ai--glm-5": "GLM 5", "minimax--minimax-m2.5": "Minimax 2.5"}


def llm_label(llm):
    return llm_to_text.get(llm, llm)


def dataset_label(identifier):
    mapping = {
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
    return mapping.get(identifier, identifier)


def fisher_correlation_ci(r, n, confidence=0.95):
    """Fisher z-transform confidence interval for a Pearson correlation coefficient."""
    if n < 4:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - (1 - confidence) / 2)
    lo, hi = z - z_crit * se, z + z_crit * se
    return np.tanh(lo), np.tanh(hi)


def correlation_table_to_latex(rows, file_path):
    column_spec = "c" * len(rows)
    llm_header = " & ".join(row["llm"] for row in rows)
    r_row = " & ".join(f"{row['correlation']:.3f}" for row in rows)
    ci_row = " & ".join(f"[{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]" for row in rows)
    n_row = " & ".join(str(row["n"]) for row in rows)

    latex_output = [
        f"\\begin{{tabular}}{{l{column_spec}}}",
        "    \\toprule",
        f"     & {llm_header} \\\\",
        "    \\midrule",
        f"    $n$ & {n_row} \\\\",
        f"    Pearson $r$ & {r_row} \\\\",
        f"    95\\% CI & {ci_row} \\\\",
        "    \\bottomrule",
        "\\end{tabular}",
    ]

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_output))
    print(f"Successfully saved table to {file_path}")


def graph_to_adjacency_matrix(graph, n_nodes):
    adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    for source, target in graph.edges():
        adjacency_matrix[source, target] = -1
        adjacency_matrix[target, source] = 1
    return adjacency_matrix


def main():
    with open("results/main_evaluation/_configs/_best_config_tag_pc_0_on_ges.json", "r") as f:
        best_config = json.load(f)
    anti_tags = best_config["anti_tags"]
    remove_duplicates = best_config["remove_duplicates"]
    remove_singular_tags = best_config["remove_singular_tags"]
    experimental_series = "tag_consistency"
    identifiers = [
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
    seed = 0 # we use the true CPDAG and tags are always the same, so one seed suffices

    path = f"E_results/E_1_tag_consistency/{experimental_series}"
    os.makedirs(path, exist_ok=True)

    correlation_rows = []

    for llm in llms:
        point_data = []

        for identifier in identifiers:
            print(f"Running {identifier}...")

            # make deterministic
            util.make_deterministic(seed)

            # load dataset
            variables, var_labels, tags, edges, positions, samples = util.load_data(identifier, 10000, order_data="random", seed=seed)

            if identifier.startswith("bnlearn"):
                data_id = identifier.split("_")[1]
            else:
                data_id = identifier
            load_str = f"{llm}__tag__{data_id}___0.txt"
            tags = util.load_from_llm(
                load_str,
                variables=var_labels,
                anti_tags=anti_tags,
                remove_duplicates=remove_duplicates,
                remove_singular_tags=remove_singular_tags,
            )
            tag_list = [tags[var] for var in var_labels]
            tags = tag_list

            # create ANM object
            anm = util.create_model(variables, edges, seed)
            true_graph = anm.graph

            # sample from ANM
            if samples is None:
                raise ValueError("Samples are None. Please provide a valid dataset or generate samples from the ANM.")
            else:
                # set all edge weights to 1, that is not correct but it might avoid confusion
                # (as the ANM weights are never actually used, and the adjacency matrix is only about edge presence)
                for u, v, data in true_graph.edges(data=True):
                    data['weight'] = 1

            # get and visualize ground truth skeleton
            _, skeleton_with_v = util.get_skeleton(true_graph)

            # apply PC on ground truth skeleton with v-structures (i.e., just apply meeks)
            gt_skeleton = util.skeleton_to_cg_graph(skeleton_with_v, undirected=False).G.graph
            cg_skel_v_meeks, _ = pc(samples, indep_test="chisq", alpha=0.05, gt_skeleton=gt_skeleton)

            ##################################################
            # Code from tagging algorithm to get tag evidences
            ##################################################

            # get unique tags
            unique_tags = list(set([tag for var_tags in tags for tag in var_tags]))
            unique_tags.sort()

            graph = cg_skel_v_meeks.G.graph

            # get directed and undirected edges
            directed_edges, undirected_edges = get_edges(graph)

            # get nr of edges per tag combination
            edge_count = get_edge_count(unique_tags, tags, directed_edges)

            directed_edges_consistencies = {}
            for tag_pair in edge_count:
                tag_a = tag_pair[0]
                tag_b = tag_pair[1]
                # ignore tag pairs with no evidence
                if edge_count[(tag_a, tag_b)] + edge_count[(tag_b, tag_a)] == 0:
                    continue
                # ignore tag pairs with same tags
                if tag_a == tag_b:
                    continue
                # only consider tag pairs once
                if tag_a > tag_b:
                    continue
                consistency_score = edge_count[(tag_a, tag_b)] / (edge_count[(tag_a, tag_b)] + edge_count[(tag_b, tag_a)])
                directed_edges_consistencies[(tag_a, tag_b)] = consistency_score

            graph = graph_to_adjacency_matrix(true_graph, len(variables))

            # get directed and undirected edges (the true graph is fully directed)
            all_edges, _ = get_edges(graph)

            # restrict to true-graph edges whose skeleton PC left undirected
            undirected_pairs = {frozenset(edge) for edge in undirected_edges}
            new_edges = [edge for edge in all_edges if frozenset(edge) in undirected_pairs]

            # get nr of edges per tag combination
            edge_count = get_edge_count(unique_tags, tags, new_edges)

            new_edges_consistency = {}
            # now get the consistency score for the new edge set
            for tag_pair in directed_edges_consistencies:
                tag_a = tag_pair[0]
                tag_b = tag_pair[1]
                denominator = edge_count[(tag_a, tag_b)] + edge_count[(tag_b, tag_a)]
                # in "undirected_only" mode, a tag pair may have no evidence left
                if denominator == 0:
                    continue
                consistency_score = edge_count[(tag_a, tag_b)] / denominator
                new_edges_consistency[(tag_a, tag_b)] = consistency_score

            for tag_pair, directed_score in directed_edges_consistencies.items():
                if tag_pair not in new_edges_consistency:
                    continue
                point_data.append(
                    {
                        "dataset": dataset_label(identifier),
                        "tag_pair": [tag_pair[0], tag_pair[1]],
                        "directed_score": directed_score,
                        "new_score": new_edges_consistency[tag_pair],
                    }
                )

        directed_scores = np.array([p["directed_score"] for p in point_data])
        comparison_scores = np.array([p["new_score"] for p in point_data])
        correlation = np.corrcoef(directed_scores, comparison_scores)[0, 1] if len(directed_scores) > 1 else float("nan")
        ci_lo, ci_hi = fisher_correlation_ci(correlation, len(directed_scores))
        print(f"{llm_label(llm)}): Pearson correlation = {correlation:.3f}, "
              f"95% CI (Fisher) = [{ci_lo:.3f}, {ci_hi:.3f}] (n={len(directed_scores)})")
        correlation_rows.append({
            "llm": llm_label(llm),
            "correlation": correlation,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n": len(directed_scores),
        })

    correlation_table_to_latex(correlation_rows, f"{path}/correlation_table.txt")


if __name__ == "__main__":
    main()
