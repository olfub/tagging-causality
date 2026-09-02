import numpy as np
import util
import os

datasets = ["bnlearn_child", "bnlearn_earthquake", "bnlearn_insurance", "bnlearn_survey", "bnlearn_asia", "bnlearn_cancer", "bnlearn_alarm", "lucas", "bnlearn_hepar2", "bnlearn_win95pts", "bnlearn_hailfinder"]
seeds = list(range(10))
path = "results/final_eval"

configs = (("pc", "ges", "true_skel"))
for config in configs:
    for dataset in datasets:
        undirected_edges = set()
        for seed in seeds:
            variables, var_labels, tags, edges, positions, samples = util.load_data(dataset, 10000, order_data="random", seed=seed)
            if config in ["pc", "true_skel"]:
                file = f"results/final_eval/{dataset}/random/{seed}/chisq_0.05_10000/_pc/adjacency_matrices.npy"
            else:
                assert config == "ges"
                file = f"results/final_eval/{dataset}/random/{seed}/chisq_0.05_10000/_ges/adjacency_matrices.npy"
            adjacency_matrices = np.load(file)
            if config == "pc":
                adj = adjacency_matrices[3]
            elif config == "true_skel":
                adj = adjacency_matrices[4]
            elif config == "ges":
                adj = adjacency_matrices[5]
            for i in range(len(adj)):
                for j in range(len(adj)):
                    if adj[i][j] == 1 and adj[j][i] == 1:
                        var_1 = var_labels[i]
                        var_2 = var_labels[j]
                        var_pair = tuple(sorted((var_1, var_2)))
                        undirected_edges.add(var_pair)
        output_dir = f"llm_edge_prediction/{config}/"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{dataset}.txt"
        with open(output_file, 'w') as f:
            for edge in undirected_edges:
                f.write(f"{edge[0]} -- {edge[1]}\n")