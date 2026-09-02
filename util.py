import csv
import json
import os
import pickle

import networkx as nx
import numpy as np
import torch
from causallearn.graph.GraphClass import CausalGraph
from networkx.drawing.nx_agraph import to_agraph

from data.bnlearn_data import get_data
from data.data import load_lucas
from generate_data import ANM
import random


def make_deterministic(seed=0):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def create_model(variables, edges=None, seed=0):
    anm = ANM(seed=seed)
    if edges is not None:
        anm.generate_new_dag(len(variables), len(edges), graph=(variables, edges))
    else:
        raise ValueError("Edges must be provided")  # TODO
    return anm


def get_directed_edges(adjacency_matrix):
    # input adjacency matrix where 1 indicates an edge from j to i, -1 indicates an edge from i to j, and 0 indicates no edge
    assert np.all(np.isin(adjacency_matrix, [-1, 0, 1]))
    edges = []
    num_nodes = adjacency_matrix.shape[0]

    # Iterate through the adjacency matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i, j] == 1:
                if (j, i) not in edges:
                    edges.append((j, i))  # Edge from j to i
            elif adjacency_matrix[i, j] == -1:
                if (i, j) not in edges:
                    edges.append((i, j))  # Edge from i to j

    return edges


def visualize_graph(graph, positions=None, labels=None, name=None):
    A = to_agraph(graph)
    # remove duplicate edges
    to_remove = [edge for edge in A.edges() if (edge[1], edge[0]) in A.edges()]
    A.remove_edges_from(to_remove)
    # add removed edges as undirected edges
    to_add = [edge for edge in to_remove if edge[0] < edge[1]]
    A.add_edges_from(to_add, dir="none")
    for count, node in enumerate(A.nodes()):
        if labels:
            node.attr["label"] = labels[count]
        if positions:
            node.attr["pos"] = f"{positions[count][0]},{positions[count][1]}!"
    A.graph_attr["splines"] = "true"
    A.draw(f"{name}.pdf", prog="neato" if positions else "dot")


def load_data(dataset_ident, nr_samples=10000, order_data="default", seed=0):
    if dataset_ident.startswith("bnlearn"):
        variables, var_labels, tags, edges, positions, data = get_data(dataset_ident.split("_")[1], nr_samples=nr_samples, seed=seed)
    elif dataset_ident == "lucas":
        variables, var_labels, tags, edges, positions, data = load_lucas()
    else:
        raise ValueError(f"Unknown dataset identifier: {dataset_ident}")
    
    if order_data == "default":
        pass
    elif order_data == "random":
        rng = np.random.default_rng(seed=int(seed))
        indices = rng.permutation(len(variables))
        variables = [variables[i] for i in indices]
        var_labels = [var_labels[i] for i in indices]
        data = data[:, indices]
    elif order_data == "invert":
        variables = variables[::-1]
        var_labels = var_labels[::-1]
        data = np.flip(data, axis=1)
    else:
        raise ValueError(f"Unknown order_data value: {order_data}")

    return variables, var_labels, tags, edges, positions, data


def get_skeleton(graph):
    orig_adj = nx.adjacency_matrix(graph).todense()
    adj = nx.adjacency_matrix(graph).todense()
    # remove weights (set all to 1)
    orig_adj[orig_adj != 0] = 1
    # iterate over adjacency matrix
    for i in range(orig_adj.shape[0]):
        for j in range(orig_adj.shape[1]):
            # undirect all edges by setting the transpose to 1
            if orig_adj[i, j] == 1:
                adj[j, i] = 1
                adj[i, j] = 1
    adj_skel = adj.copy()
    # iterate over adjacency matrix columns
    for i in range(orig_adj.shape[1]):
        # if there are at least 2 incoming edges in the gt
        if sum(orig_adj[:, i]) > 1:
            # iterate over all incoming edges
            for j in range(orig_adj.shape[0]):
                # set incoming edges to directed (remove the edge from the transpose)
                if orig_adj[j, i] == 1:
                    adj[j, i] = 1
                    adj[i, j] = 0
    # copy adj
    adj_skel_with_v= adj.copy()
    skel_graph = nx.from_numpy_array(adj_skel, create_using=nx.DiGraph())
    skel_graph_with_v = nx.from_numpy_array(adj_skel_with_v, create_using=nx.DiGraph())
    return skel_graph, skel_graph_with_v


def skeleton_to_cg_graph(skeleton, undirected):
    # turn a networkx digraph into a causalgraph object as used in causallearn
    skeleton_adj = nx.adjacency_matrix(skeleton).todense()
    cg_adj = nx.adjacency_matrix(skeleton).todense()
    cg_adj[:] = 0
    for i in range(skeleton_adj.shape[0]):
        for j in range(skeleton_adj.shape[1]):
            if skeleton_adj[i, j] == 1:
                if skeleton_adj[j, i] == 0:
                    # directed edge from i to j
                    cg_adj[i, j] = -1
                    # if undirected, then only insert all edges as undirected
                    cg_adj[j, i] = -1 if undirected else 1
                else:
                    # undirected edge
                    cg_adj[i, j] = -1
                    cg_adj[j, i] = -1
    cg_object = CausalGraph(len(skeleton.nodes), None)
    cg_object.G.graph = cg_adj.astype(int)
    return cg_object


def cg_to_nx(cg):
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(range(cg.G.graph.shape[0]))
    nx_graph.add_edges_from(get_directed_edges(cg.G.graph))
    return nx_graph


def ges_to_nx(ges):
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(range(ges.graph.shape[0]))
    nx_graph.add_edges_from(get_directed_edges(ges.graph))
    return nx_graph


def get_directed_edges_v2(adjacency_matrix):
    # this version takes as input adjacency_matrices where each values that is not 0 is a directed edge from j to i
    # with the value being the weight of the edge (but that is ignored here)
    edges = []
    num_nodes = adjacency_matrix.shape[0]

    # Iterate through the adjacency matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i, j] != 0:
                edges.append((i, j))  # Edge from j to i

    return edges

def array_to_nx(array):
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(range(array.shape[0]))
    nx_graph.add_edges_from(get_directed_edges_v2(array))
    return nx_graph


def create_path(experimental_series, args):
    path = f"results/{experimental_series}/{args.identifier}/{args.order_data}/{args.seed}/{args.pc_indep_test}_{args.pc_alpha}_{args.ges_score_func}_{args.nr_samples}"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def create_path_and_id(experimental_series, args):
    path = f"results/{experimental_series}/"
    if not os.path.exists(path):
        os.makedirs(path)
    id = f"{args.identifier}_{args.seed}_{args.tagging_approach_id}_{args.nr_samples}"
    # save arguments as json in folder
    with open(f"{path}/{id}_args.json", "w") as f:
        json.dump(vars(args), f)
    return path, id


def save_info(info, true_graph, name, variables=None, use_pickle=False):
    if use_pickle:
        with open(f"{name}_info.pkl", "wb") as f:
            pickle.dump(info, f)
    adj = nx.adjacency_matrix(true_graph).todense()
    new_info = []
    for item in info:
        edge = [item[0][0], item[0][1]]
        if adj[edge[0], edge[1]] == 0:
            pred = 0
        else:
            pred = 1
        if variables is not None:
            new_info.append([pred, [variables[item[0][0]], variables[item[0][1]]], item[1], item[2], item[3], item[4], item[5]])
            
    with open(f"{name}_info.csv", "w") as f:
        writer = csv.writer(f)
        for item in new_info:
            writer.writerow(item)

def load_info_pickle(name):
    with open(f"{name}_info.pkl", "rb") as f:
        info = pickle.load(f)
    return info

def save_eval(eval_scores, path, use_pickle=False):
    if use_pickle:
        with open(f"{path}/eval.pkl", "wb") as f:
            pickle.dump(eval_scores, f)
    with open(f"{path}/eval.csv", "w") as f:
        writer = csv.writer(f)
        for es in eval_scores:
            writer.writerow(es)

def save_adjacency_matrices(adjacency_matrices, path):
    all_adj = np.stack(adjacency_matrices)
    np.save(f"{path}/adjacency_matrices.npy", all_adj)

def save_names(names, path):
    with open(f"{path}/names.txt", "w") as f:
        for name in names:
            f.write(f"{name}\n")

def save_all(name, infos, info_names, true_graph, eval_scores, adjacency_matrices, names, variables=None, args=None):
    for info, info_name in zip(infos, info_names):
        save_info(info, true_graph, f"{name}/{info_name}", variables)
    save_eval(eval_scores, name)
    save_adjacency_matrices(adjacency_matrices, name)
    save_names(names, name)
    if args is not None:
        with open(f"{name}/args.json", "w") as f:
            json.dump(vars(args), f)

def save_pc_res(pc_folder, cg, cg_1, skeleton_with_v, pc_skeleton, pc_sepset, adjacency_matrices, eval_scores, names):
    # save cg
    with open(f"{pc_folder}/cg.pkl", "wb") as f:
        pickle.dump(cg, f)
    # save cg_1
    with open(f"{pc_folder}/cg_1.pkl", "wb") as f:
        pickle.dump(cg_1, f)
    # save skeleton with v
    with open(f"{pc_folder}/skeleton_with_v.pkl", "wb") as f:
        pickle.dump(skeleton_with_v, f)
    # save skeleton
    with open(f"{pc_folder}/skeleton.pkl", "wb") as f:
        pickle.dump(pc_skeleton, f)
    # save sepsets
    with open(f"{pc_folder}/sepsets.pkl", "wb") as f:
        pickle.dump(pc_sepset, f)
    # save adjacency matrices
    save_adjacency_matrices(adjacency_matrices, pc_folder)
    # save eval scores
    save_eval(eval_scores, pc_folder, use_pickle=True)
    # save names
    save_names(names, pc_folder)

def load_pc_res(pc_folder):
    # load cg
    with open(f"{pc_folder}/cg.pkl", "rb") as f:
        cg = pickle.load(f)
    # load cg_1
    with open(f"{pc_folder}/cg_1.pkl", "rb") as f:
        cg_1 = pickle.load(f)
    # load skeleton with v
    with open(f"{pc_folder}/skeleton_with_v.pkl", "rb") as f:
        skeleton_with_v = pickle.load(f)
    # load skeleton
    with open(f"{pc_folder}/skeleton.pkl", "rb") as f:
        pc_skeleton = pickle.load(f)
    # load sepsets
    with open(f"{pc_folder}/sepsets.pkl", "rb") as f:
        pc_sepset = pickle.load(f)
    # load adjacency matrices
    adjacency_matrices = list(np.load(f"{pc_folder}/adjacency_matrices.npy"))
    # load eval scores
    eval_scores = pickle.load(open(f"{pc_folder}/eval.pkl", "rb"))
    # load names
    with open(f"{pc_folder}/names.txt", "r") as f:
        names = [line.strip() for line in f]
    return cg, cg_1, skeleton_with_v, pc_skeleton, pc_sepset, adjacency_matrices, eval_scores, names

def save_ges_res(ges_folder, graph_ges, adjacency_matrices, eval_scores, names):
    # save graph_ges
    with open(f"{ges_folder}/graph_ges.pkl", "wb") as f:
        pickle.dump(graph_ges, f)
    # save adjacency matrices
    save_adjacency_matrices(adjacency_matrices, ges_folder)
    # save eval scores
    save_eval(eval_scores, ges_folder, use_pickle=True)
    # save names
    save_names(names, ges_folder)

def load_ges_res(ges_folder):
    # load graph_ges
    with open(f"{ges_folder}/graph_ges.pkl", "rb") as f:
        graph_ges = pickle.load(f)
    # load adjacency matrices
    adjacency_matrices = list(np.load(f"{ges_folder}/adjacency_matrices.npy"))
    # load eval scores
    eval_scores = pickle.load(open(f"{ges_folder}/eval.pkl", "rb"))
    # load names
    with open(f"{ges_folder}/names.txt", "r") as f:
        names = [line.strip() for line in f]
    return graph_ges, adjacency_matrices, eval_scores, names

def save_adj_eval(folder, adjacency_matrices, eval_scores, names):
    # save adjacency matrices
    save_adjacency_matrices(adjacency_matrices, folder)
    # save eval scores
    save_eval(eval_scores, folder, use_pickle=True)
    # save names
    save_names(names, folder)

def load_adj_eval(folder):
    # load adjacency matrices
    adjacency_matrices = list(np.load(f"{folder}/adjacency_matrices.npy"))
    # load eval scores
    eval_scores = pickle.load(open(f"{folder}/eval.pkl", "rb"))
    # load names
    with open(f"{folder}/names.txt", "r") as f:
        names = [line.strip() for line in f]
    return adjacency_matrices, eval_scores, names

def prep_sep_sets_for_typing(sep_sets):
    new_sep_sets = np.empty(sep_sets.shape, dtype=object)
    for i in range(new_sep_sets.shape[0]):
        for j in range(new_sep_sets.shape[1]):
            this_set = set()
            if sep_sets[i, j] is not None:
                for k in range(len(sep_sets[i, j])):
                    this_set = this_set.union(sep_sets[i, j][k])
            new_sep_sets[i, j] = this_set
    return new_sep_sets

def load_from_llm(file, variables, anti_tags=False, remove_duplicates=True, remove_singular_tags=True, variation=False):
    keys = {}
    path = f"queries/processed/taggingsets/{file}"
    if variation:
        path = f"queries/processed/taggingsets_variation/{file}"
    with open(path, 'r') as f:
        for line in f:
            key, value = line.strip().split(':')
            vars = value.split(',')
            keys[key] = vars
    if "type" in file:
        types = {}
        for var in variables:
            for type, type_variables in keys.items():
                if var in type_variables:
                    if var in types:
                        continue
                        # raise ValueError(f"Variable {var} is assigned to multiple types")
                    types[var] = type
            # if var not in types:
            #     raise ValueError(f"Variable {var} is not assigned to any type")
        return types
    else:
        # adding anti keys
        if anti_tags:
            keys_items = list(keys.items())
            for k, v in keys_items:
                keys["anti_" + k] = [var for var in variables if var not in v]
        if remove_duplicates:
            # Remove keys from "keys" where items already exist
            unique_keys = {}
            seen_items = set()
            for k, v in keys.items():
                if set(v) not in seen_items:
                    unique_keys[k] = v
                    seen_items.add(frozenset(v))
            keys = unique_keys

        # create dictionary of tags for each variable
        tags = {}
        for var in variables:
            tags[var] = []
            for tag, tag_variables in keys.items():
                # remove all tags that only have one variable
                if remove_singular_tags and len(tag_variables) < 2:
                    continue
                if var in tag_variables:
                    tags[var].append(tag)
        return tags
    

def introduce_tag_errors(tags, tag_list, var_labels, error_rate=0.1):
    all_vars = var_labels
    all_tags = [tag for var in tag_list for tag in var]
    number_tags = len(all_tags)
    number_tags_error = int(number_tags * error_rate)
    if number_tags_error % 2 == 0:
        number_tags_remove = number_tags_error // 2
        number_tags_add = number_tags_error // 2
    else:
        number_tags_remove = number_tags_error // 2
        number_tags_add = number_tags_error // 2 + 1
    
    # add incorrect tags
    unique_tags = set(all_tags)
    added = 0
    new_tags = {var: list(tags[var]) for var in tags}
    while added < number_tags_add:
        var = random.choice(all_vars)
        possible_tags = list(unique_tags - set(new_tags[var]))
        if possible_tags:
            new_tag = random.choice(possible_tags)
            new_tags[var].append(new_tag)
            added += 1

    # remove correct tags
    removed = 0
    new_tags_removed = {var: list(tags[var]) for var in tags}
    removed_tags = {var: [] for var in tags}
    while removed < number_tags_remove:
        weights = [len(new_tags_removed[var]) for var in all_vars]
        var = random.choices(all_vars, weights=weights, k=1)[0]
        if new_tags_removed[var]:
            tag_to_remove = random.choice(new_tags_removed[var])
            new_tags_removed[var].remove(tag_to_remove)
            removed_tags[var].append(tag_to_remove)
            removed += 1

    # combine into the new tags
    new_tags_combined = {}
    for var in tags:
        new_tags_combined[var] = [tag for tag in new_tags[var] if tag not in removed_tags[var]]
    
    # into tag_list
    new_tag_list = [new_tags_combined[var] for var in var_labels]
    if number_tags_error % 2 == 0:
        assert number_tags == len([tag for var in new_tag_list for tag in var])
    else:
        assert number_tags == len([tag for var in new_tag_list for tag in var]) - 1
    return new_tag_list


def introduce_type_errors(types, var_labels, error_rate=0.1):
    # assign new types for variables that don't have one
    vars_without_type = [var for var in var_labels if var not in types]
    for var in vars_without_type:
        types[var] = f"personal_type_{var}"

    all_vars = var_labels
    unique_types = set(types.values())
    number_types = len(all_vars)
    number_types_error = int(number_types * error_rate)
    
    changed = 0
    new_types = {var: types[var] for var in types}
    vars_changed = set()
    while changed < number_types_error:
        var = random.choice(list(set(all_vars) - vars_changed))
        possible_types = list(unique_types - {types[var]})
        if possible_types:
            new_type = random.choice(possible_types)
            new_types[var] = new_type
            changed += 1
            vars_changed.add(var)

    return new_types


def remove_random_tags(tag_list, fewer_tags):
    unique_tags = set(tag for var in tag_list for tag in var)
    number_tags = len(unique_tags)
    number_tags_remove = int(number_tags * fewer_tags)
    remove_tags = set()
    for i in range(number_tags_remove):
        tag = random.choice(list(unique_tags - remove_tags))
        remove_tags.add(tag)
    new_tags = unique_tags - remove_tags
    new_tag_list = []
    for tags in tag_list:
        new_tags_for_var = [tag for tag in tags if tag in new_tags]
        new_tag_list.append(new_tags_for_var)    

    return new_tag_list