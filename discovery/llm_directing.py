import numpy as np
import json
import util

def apply_edge_directions(llm, identifier, graph, var_labels):
    undirected_edges = set()
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i][j] == 1 and graph[j][i] == 1:
                var_1 = var_labels[i]
                var_2 = var_labels[j]
                var_pair = tuple(sorted((var_1, var_2)))
                undirected_edges.add(var_pair)

    # use llm to direct edges
    data_tag = identifier if "bnlearn" not in identifier else identifier.split("_")[1]
    with open(f"queries/processed/pairwise_completion/{llm}_{data_tag}.json", "r") as file:
        llm_edges = json.load(file)
        direct_edges = []
        for edge, direction in llm_edges.items():
            edge_a = edge.split("__")[0]
            edge_b = edge.split("__")[1]
            if direction == "AB":
                direct_edges.append((edge_a, edge_b))
            elif direction == "BA":
                direct_edges.append((edge_b, edge_a))
            else:
                raise ValueError(f"Unexpected direction {direction} for edge {edge}")
    with open(f"queries/processed/pairwise_completion/{llm}_{data_tag}_missing.json", "r") as file:
        llm_edges_missing = json.load(file)
        missing_edges = []
        for edge in llm_edges_missing:
            edge_a = edge.split("__")[0]
            edge_b = edge.split("__")[1]
            missing_edges.append((edge_a, edge_b))
            missing_edges.append((edge_b, edge_a))

    # update edges
    new_adj = np.copy(graph)
    for edge in undirected_edges:
        if edge in direct_edges:
            edge_a_idx = var_labels.index(edge[0])
            edge_b_idx = var_labels.index(edge[1])
            new_adj[edge_a_idx][edge_b_idx] = 1
            new_adj[edge_b_idx][edge_a_idx] = 0
        elif (edge[1], edge[0]) in direct_edges:
            edge_a_idx = var_labels.index(edge[0])
            edge_b_idx = var_labels.index(edge[1])
            new_adj[edge_a_idx][edge_b_idx] = 0
            new_adj[edge_b_idx][edge_a_idx] = 1
        else:
            assert edge in missing_edges, f"Edge {edge} not found in direct edges or missing edges"

    # evaluate
    graph = util.array_to_nx(new_adj)
    return graph


def load_llm_predictions(llm, identifier, name, var_labels, labels):
    data_tag = identifier if "bnlearn" not in identifier else identifier.split("_")[1]
    if name == "llm_root_recursive":
        open_str_adj = f"queries/processed/root_recursive/{llm}_root_recursive_{data_tag}_adj.txt"
        open_str_labels = f"queries/processed/root_recursive/{llm}_root_recursive_{data_tag}_adj_var_names.txt"
    else:
        raise ValueError(f"Unexpected name {name}")
    with open(open_str_adj, "r") as file:
        adj_array = np.asarray(json.load(file))

    # load variables order (only stored in full_pairwise, same for root_recursive)
    with open(open_str_labels, "r") as file:
        var_order = json.load(file)

    assert len(var_order) == len(var_labels) and all(var in var_order for var in var_labels), f"Not the same variables in the llm predictions as in the data for {data_tag}"

    assert adj_array.shape[0] == len(var_order) and adj_array.shape[1] == len(var_order), f"Adjacency matrix shape does not match variable order length in {data_tag}"

    new_adj = np.zeros_like(adj_array)
    for index, var in labels.items():
        for index2, var2 in labels.items():
            source_index = var_order.index(var)
            source_index2 = var_order.index(var2)
            new_adj[index][index2] = adj_array[source_index][source_index2]

    # evaluate
    graph = util.array_to_nx(new_adj)
    return graph