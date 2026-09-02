# copied and adapted from causal-learn package

from __future__ import annotations

import time
import warnings
from itertools import combinations, permutations
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.cit import *
from causallearn.utils.PCUtils import Helper, Meek, SkeletonDiscovery, UCSepset
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import (
    orient_by_background_knowledge,
)
from numpy import ndarray
from copy import deepcopy


def pc_tagged(
    algorithm: int,
    data: ndarray,
    tags: Dict[str, List[str]],
    alpha=0.05, 
    indep_test=fisherz, 
    stable: bool = True, 
    uc_rule: int = 0, 
    uc_priority: int = 2,
    min_samples: int = 2,
    min_prob_threshold: float = 0.5,
    background_knowledge: BackgroundKnowledge | None = None, 
    gt_skeleton: ndarray | None = None,
    search_v_structures: bool = True,
    verbose: bool = False, 
    show_progress: bool = True,
    node_names: List[str] | None = None,
    prior_on_weight: bool = False,
    always_meeks: bool = False,
    redirect_existing_edges: bool = True,
    redirecting_strategy: int = 0,
    min_prob_redirecting: float = 0.6,
    include_current_edge_as_evidence: bool = False,
    include_redirected_edges_in_edge_count: bool = True,
    cg: CausalGraph | None = None,
    force_tagging: bool = False,
    laplace: bool = False,
    **kwargs
):
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    cg_new = deepcopy(cg)

    if algorithm == 0:
        return pc_tagged_alg(data=data, tags=tags, node_names=node_names, alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule,
                        uc_priority=uc_priority, min_samples=min_samples, min_prob_threshold=min_prob_threshold, min_prob_redirecting=min_prob_redirecting, background_knowledge=background_knowledge,
                        gt_skeleton=gt_skeleton, search_v_structures=search_v_structures,verbose=verbose, show_progress=show_progress, prior_on_weight=prior_on_weight, always_meeks=always_meeks, 
                        redirect_existing_edges=redirect_existing_edges, redirecting_strategy=redirecting_strategy, include_current_edge_as_evidence=include_current_edge_as_evidence, 
                        include_redirected_edges_in_edge_count=include_redirected_edges_in_edge_count, cg=cg_new, force_tagging=force_tagging, laplace=laplace, **kwargs)
    elif algorithm == 1:
        return pc_tagged_alt(data=data, tags=tags, node_names=node_names, alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule, uc_priority=uc_priority, min_samples=min_samples,
                             min_prob_threshold=min_prob_threshold, background_knowledge=background_knowledge, gt_skeleton=gt_skeleton, search_v_structures=search_v_structures,verbose=verbose,
                             show_progress=show_progress, prior_on_weight=prior_on_weight, cg_1=cg_new, **kwargs)


def get_edge_count(unique_tags, tags, directed_edges, laplace=False):
    if laplace:
        edge_count = {(tag1, tag2): 1 for tag1 in unique_tags for tag2 in unique_tags}
    else:
        edge_count = {(tag1, tag2): 0 for tag1 in unique_tags for tag2 in unique_tags}
    for edge in directed_edges:
        for tag1 in tags[edge[0]]:
            for tag2 in tags[edge[1]]:
                if tag1 == tag2:
                    continue
                edge_count[(tag1, tag2)] += 1
    return edge_count


def get_edges(graph):
    directed_edges = []
    undirected_edges = []
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph[i, j] == -1 and graph[j, i] == 1:
                directed_edges.append((i, j))
            elif graph[i, j] == -1 and graph[j, i] == -1:
                undirected_edges.append((i, j))
    return directed_edges, undirected_edges


def find_best_to_direct(edges, tags, edge_count, min_samples, backward_only=False, include_current_edge_as_evidence=True, var_tags=None, laplace=False):
    pseudo_count = 1 if laplace else 0
    best_edge_score_direction = (None, 0, True)
    for edge in edges:
        if tags[edge[0]] == tags[edge[1]]:
            continue
        forward_score = 0
        backward_score = 0
        fw_evidence = 0
        bw_evidence = 0
        forward_tag_pairs = []
        backward_tag_pairs = []
        for tag1 in tags[edge[0]]:
            for tag2 in tags[edge[1]]:
                if tag1 == tag2:
                    continue
                fw = edge_count[(tag1, tag2)]
                if not include_current_edge_as_evidence:
                    fw -= 1  # since this forward direction already was used as evidence into edge_count
                if fw > 0:
                    if var_tags is None:
                        forward_score += fw
                    else:
                        forward_score += fw / (2**(len(var_tags[tag1]) + len(var_tags[tag2]) -1))
                    if fw - pseudo_count > 0:
                        fw_evidence += fw - pseudo_count
                    forward_tag_pairs.append((tag1, tag2, fw))
                bw = edge_count[(tag2, tag1)]
                if bw > 0:
                    if var_tags is None:
                        backward_score += bw
                    else:
                        backward_score += bw / (2**(len(var_tags[tag1]) + len(var_tags[tag2]) -1))
                    if bw - pseudo_count > 0:
                        bw_evidence += bw - pseudo_count
                    backward_tag_pairs.append((tag2, tag1, bw))
        # determine probability for directing this edge in both directions
        if fw_evidence + bw_evidence < min_samples:
            forward_prob = 0.5
            backward_prob = 0.5
        else:
            forward_prob = forward_score / (forward_score + backward_score)
            backward_prob = 1 - forward_prob

        if not backward_only:
            if forward_prob > best_edge_score_direction[1]:
                best_edge_score_direction = (edge, forward_prob, True, (forward_tag_pairs, backward_tag_pairs))
        if backward_prob > best_edge_score_direction[1]:
            best_edge_score_direction = (edge, backward_prob, False, (forward_tag_pairs, backward_tag_pairs))
    return best_edge_score_direction


def find_best_threshold(graph, tags, min_samples, prior_on_weight, laplace=False):
    num_nodes = graph.shape[0]
    # get directed and undirected edges
    directed_edges, _ = get_edges(graph)
    if directed_edges == []:
        return 0.5
    # get unique tags
    unique_tags = list(set([tag for var_tags in tags for tag in var_tags]))
    if prior_on_weight:
        # number of nodes per tag
        var_tags = {}
        for i in range(num_nodes):
            for tag in unique_tags:
                if tag in tags[i]:
                    if tag in var_tags:
                        var_tags[tag].append(i)
                    else:
                        var_tags[tag] = [i]
    else:
        # of course, the number of variables per tag does not change if we don't use prior_on_weight
        # but we need to set var_tags to None to avoid using it in the following
        var_tags = None
    prob_and_correctness = []
    for current_edge in directed_edges:
        known_edges = [e for e in directed_edges if e != current_edge]  # remove the current edge from the directed edges, as we want to test both directions of this edge
        edge_count = get_edge_count(unique_tags, tags, known_edges, laplace=laplace)
        best_edge_score_direction = find_best_to_direct([current_edge], tags, edge_count, min_samples, backward_only=False, include_current_edge_as_evidence=True, var_tags=var_tags, laplace=laplace)
        prob_and_correctness.append((best_edge_score_direction[1], best_edge_score_direction[2]))
    probs = [max(0.5, pc[0]) for pc in prob_and_correctness]
    actuals = [pc[1] for pc in prob_and_correctness]
    threshold = find_optimal_threshold(probs, actuals)
    return threshold


def find_optimal_threshold(probs, actuals):
    probs = np.array(probs)
    actuals = np.array(actuals).astype(bool)
    
    # Thresholds to check: every unique p value
    threshold_candidates = np.sort(np.unique(probs))
    
    results = []
    
    for T in threshold_candidates:
        total_loss = 0
        for p, y in zip(probs, actuals):
            if p == 0.5:
                total_loss += 0.5
            elif p >= T:
                total_loss += 0.0 if y else 1.0
            else:
                total_loss += 0.5
        
        results.append(total_loss)
    
    best_idx = np.argmin(results) if len(results) > 0 and results.count(min(results)) == 1 else max([i for i, v in enumerate(results) if v == min(results)])
    return threshold_candidates[best_idx]


def pc_tagged_alg(
    data: ndarray,
    tags: Dict[str, List[str]],
    node_names: List[str] | None,
    alpha: float,
    indep_test: str,
    stable: bool,
    uc_rule: int,
    uc_priority: int,
    min_samples: int,
    min_prob_threshold: float,
    min_prob_redirecting: float,
    background_knowledge: BackgroundKnowledge | None = None,
    gt_skeleton: ndarray | None = None,
    search_v_structures: bool = True,
    verbose: bool = False,
    show_progress: bool = True,
    always_meeks: bool = False,
    redirect_existing_edges: bool = True,
    redirecting_strategy: int = 0,
    include_current_edge_as_evidence: bool = False,
    include_redirected_edges_in_edge_count: bool = True,
    prior_on_weight: bool = False,
    cg: CausalGraph | None = None,
    force_tagging: bool = False,
    laplace: bool = False,
    **kwargs
) -> CausalGraph:
    """
    Perform Peter-Clark (PC) algorithm for causal discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)
    alpha : float, desired significance level of independence tests (p_value) in (0, 1)
    indep_test : str, the name of the independence test being used
            ["fisherz", "chisq", "gsq", "kci"]
           - "fisherz": Fisher's Z conditional independence test
           - "chisq": Chi-squared conditional independence test
           - "gsq": G-squared conditional independence test
           - "kci": Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    uc_rule : how unshielded colliders are oriented
           0: run uc_sepset
           1: run maxP
           2: run definiteMaxP
    uc_priority : rule of resolving conflicts between unshielded colliders
           -1: whatever is default in uc_rule
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """
    # force_tagging: only do tagging and do it whenever the prob threshold is met (no meeks or cylicity check)
    if always_meeks and force_tagging:
        raise ValueError("always_meeks and skip_all_meeks can not be True at the same time")
    if redirect_existing_edges and force_tagging:
        raise ValueError("redirect_existing_edges and skip_all_meeks can not be True at the same time")
    if gt_skeleton is not None and search_v_structures:
        raise NotImplementedError("Supplying gt_skeleton but still obtaining v-strutures is not implemented yet")
    start = time.time()
    # first, try to use gt_skeleton if available
    if gt_skeleton is not None and not search_v_structures:
        indep_test = CIT(data, indep_test, **kwargs)
        cg = CausalGraph(data.shape[1], node_names)
        cg.set_ind_test(indep_test)
        cg.G.graph = gt_skeleton
    # if it isn't, either use cg or run the whole PC algorithm
    elif cg is None:
        indep_test = CIT(data, indep_test, **kwargs)
        cg_1 = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, stable,
                                                    background_knowledge=background_knowledge, verbose=verbose,
                                                    show_progress=show_progress, node_names=node_names)

        if background_knowledge is not None:
            orient_by_background_knowledge(cg_1, background_knowledge)

        if uc_rule == 0:
            if uc_priority != -1:
                cg_2 = UCSepset.uc_sepset(cg_1, uc_priority, background_knowledge=background_knowledge)
            else:
                cg_2 = UCSepset.uc_sepset(cg_1, background_knowledge=background_knowledge)
            cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

        elif uc_rule == 1:
            if uc_priority != -1:
                cg_2 = UCSepset.maxp(cg_1, uc_priority, background_knowledge=background_knowledge)
            else:
                cg_2 = UCSepset.maxp(cg_1, background_knowledge=background_knowledge)
            cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

        elif uc_rule == 2:
            if uc_priority != -1:
                cg_2 = UCSepset.definite_maxp(cg_1, alpha, uc_priority, background_knowledge=background_knowledge)
            else:
                cg_2 = UCSepset.definite_maxp(cg_1, alpha, background_knowledge=background_knowledge)
            cg_before = Meek.definite_meek(cg_2, background_knowledge=background_knowledge)
            cg = Meek.meek(cg_before, background_knowledge=background_knowledge)
        else:
            raise ValueError("uc_rule should be in [0, 1, 2]")
    else:
        cg = cg

    # get unique tags
    unique_tags = list(set([tag for var_tags in tags for tag in var_tags]))
    if prior_on_weight:
        # number of nodes per tag
        var_tags = {}
        for i, _ in enumerate(cg.G.nodes):
            for tag in unique_tags:
                if tag in tags[i]:
                    if tag in var_tags:
                        var_tags[tag].append(i)
                    else:
                        var_tags[tag] = [i]
    else:
        # of course, the number of variables per tag does not change if we don't use prior_on_weight
        # but we need to set var_tags to None to avoid using it in the following
        var_tags = None

    if not force_tagging:
        cg_new = Meek.meek(cg, background_knowledge=background_knowledge)
        cg = cg_new

    # cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i --> j ,
    # cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
    # cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.
    graph = cg.G.graph

    # get directed and undirected edges
    directed_edges, undirected_edges = get_edges(graph)

    if len(undirected_edges) == 0:
        end = time.time()
        cg.PC_elapsed = end - start
        return cg, []
    
    # get nr of edges per tag combination
    edge_count = get_edge_count(unique_tags, tags, directed_edges, laplace=laplace)

    directed_edge_info = []
    
    # a pass that also allows for redirecting existing edges
    loop = redirect_existing_edges
    count = 0
    redirected_edges = []
    max_count = 100
    old_directed_edges = directed_edges.copy()
    while loop:
        if count >= max_count:
            # relevant for redirecting_strategy == 0
            break
        loop = False
        best_edge_score_direction = find_best_to_direct(directed_edges, tags, edge_count, min_samples, backward_only=True, include_current_edge_as_evidence=include_current_edge_as_evidence, var_tags=var_tags, laplace=laplace)

        if best_edge_score_direction[1] > min_prob_redirecting:
            # after iteration, direct edge with highest score
            to_direct = best_edge_score_direction[0]
            # invert edge
            cg.G.graph[to_direct[0], to_direct[1]] *= -1
            cg.G.graph[to_direct[1], to_direct[0]] *= -1
            direction_valid = True
            if cg.G.exists_directed_cycle():
                # invert edge
                cg.G.graph[to_direct[0], to_direct[1]] *= -1
                cg.G.graph[to_direct[1], to_direct[0]] *= -1
                direction_valid = False
                # do not consider this directed edge and try again without it
                directed_edges.remove(to_direct)
            else:
                # update diredcted_edges (can not just add and remove one edges, as other edges that might have been removed due to cycles should be added again)
                directed_edges, undirected_edges = get_edges(cg.G.graph)

                # fully dynamically, redirect the best edge, then recalculate edge_count, then continue with all directed edges, ..., terminate at some point to avoid cycles
                if redirecting_strategy == 0:
                    # update edge count
                    edge_count = get_edge_count(unique_tags, tags, directed_edges, laplace=laplace)

                # fully data based, we calculate the evidence before redirecting, then redirect while keeping the evidence fixed
                elif redirecting_strategy == 1:
                    redirected_edges.append(to_direct)
                    redirected_edges.append((to_direct[1], to_direct[0]))
                    directed_edges = [edge for edge in directed_edges if edge not in redirected_edges]

                # hybrid, we update the evidence after each redirection, but remove the redirected edge from the evidence; we also don't allow this edge to be redirected again
                elif redirecting_strategy == 2:
                    # add to redirected edges (just adding both to be sure, can't hurt)
                    redirected_edges.append(to_direct)
                    redirected_edges.append((to_direct[1], to_direct[0]))
                    directed_edges = [edge for edge in directed_edges if edge not in redirected_edges]
                    # update edge count
                    edge_count = get_edge_count(unique_tags, tags, directed_edges, laplace=laplace)
                    
            loop = True
            directed_edge_info.append((best_edge_score_direction[0], best_edge_score_direction[1], best_edge_score_direction[2], direction_valid, best_edge_score_direction[3], True))
            if redirecting_strategy == 0:
                count += 1

    directed_edges, undirected_edges = get_edges(cg.G.graph)

    # get nr of edges per tag combination
    if include_redirected_edges_in_edge_count:
        evidence_edges = directed_edges
    else:
        evidence_edges = [edge for edge in directed_edges if edge in old_directed_edges]
    edge_count = get_edge_count(unique_tags, tags, evidence_edges, laplace=laplace)

    # while edges are undirected
    while len(undirected_edges) > 0:
        # iterate over all undirected edges
        # remember edge with highest probability (first element is edge, second probability, third direction)
        # direction is True if edge is directed from first to second node
        best_edge_score_direction = find_best_to_direct(undirected_edges, tags, edge_count, min_samples, backward_only=False, var_tags=var_tags, laplace=laplace)

        if best_edge_score_direction[1] == 0.5 or best_edge_score_direction[1] < min_prob_threshold:
            break

        # after iteration, direct edge with highest score
        to_direct = best_edge_score_direction[0]

        if best_edge_score_direction[2]:
            cg.G.graph[to_direct[1], to_direct[0]] = 1
        else:    
            cg.G.graph[to_direct[0], to_direct[1]] = 1

        direction_valid = True
        if cg.G.exists_directed_cycle() and not force_tagging:
            # invert edge
            cg.G.graph[to_direct[0], to_direct[1]] *= -1
            cg.G.graph[to_direct[1], to_direct[0]] *= -1
            direction_valid = False
            # print(f"Cycle detected, inverting edge")

        directed_edge_info.append((best_edge_score_direction[0], best_edge_score_direction[1], best_edge_score_direction[2], direction_valid, best_edge_score_direction[3], False))

        if always_meeks:
            cg_new = Meek.meek(cg, background_knowledge=background_knowledge)
            cg = cg_new

        # get undirected edges  # TODO could do something more efficient and only update what changed but this one works as well
        directed_edges, undirected_edges = get_edges(cg.G.graph)

    if not force_tagging:
        cg_after_meek = Meek.meek(cg, background_knowledge=background_knowledge)
    else:
        cg_after_meek = cg

    end = time.time()

    cg.PC_elapsed = end - start

    return cg_after_meek, directed_edge_info


def pc_tagged_alt(
    data: ndarray,
    tags: Dict[str, List[str]],
    node_names: List[str] | None,
    alpha: float,
    indep_test: str,
    stable: bool,
    uc_rule: int,
    uc_priority: int,
    min_samples: int,
    min_prob_threshold: float,
    background_knowledge: BackgroundKnowledge | None = None,
    gt_skeleton: ndarray | None = None,
    search_v_structures: bool = True,
    verbose: bool = False,
    show_progress: bool = True,
    prior_on_weight: bool = False,
    cg_1: CausalGraph | None = None,
    **kwargs
) -> CausalGraph:
    start = time.time()
    if cg_1 is None:
        indep_test = CIT(data, indep_test, **kwargs)

        # 1. get skeleton and sep sets
        cg_1 = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, stable,
                                                    background_knowledge=background_knowledge, verbose=verbose,
                                                    show_progress=show_progress, node_names=node_names)
    else:
        cg_1 = cg_1
    
    # get unique tags
    unique_tags = list(set([tag for var_tags in tags for tag in var_tags]))
    if prior_on_weight:
        # number of nodes per tag
        var_tags = {}
        for i, _ in enumerate(cg_1.G.nodes):
            for tag in unique_tags:
                if tag in tags[i]:
                    if tag in var_tags:
                        var_tags[tag].append(i)
                    else:
                        var_tags[tag] = [i]
    else:
        # of course, the number of variables per tag does not change if we don't use prior_on_weight
        # but we need to set var_tags to None to avoid using it in the following
        var_tags = None

    # 2. collect all triple nodes (unshielded)
    triples = cg_1.find_unshielded_triples()

    # 3. try out all tag consistency possibilities and save evidence
    edge_count = {(tag1, tag2): 0 for tag1 in unique_tags for tag2 in unique_tags}
    # go over unshielded 
    for triple in triples:
        current_sep_sets = cg_1.sepset[triple[0], triple[2]]
        assert current_sep_sets is not None  # because of unshielded triplet
        if len(current_sep_sets) == 1:
            all_sep_set_vars = current_sep_sets[0]
        else:
            all_sep_set_vars = current_sep_sets[0] + current_sep_sets[1]
        v_structure_found = triple[1] not in all_sep_set_vars
        # check for v-structure
        if v_structure_found:
            # v-structure
            edge_1 = (triple[0], triple[1])
            edge_2 = (triple[2], triple[1])
            both_edges = [edge_1, edge_2]
            # for both edges
            for edge in both_edges:
                # count for normalization
                nr_tag_pairs = 0
                for tag1 in tags[edge[0]]:
                    for tag2 in tags[edge[1]]:
                        # ignore same tag cases
                        if tag1 == tag2:
                            continue
                        nr_tag_pairs += 1

                # add to tag_evidence
                for tag1 in tags[edge[0]]:
                    for tag2 in tags[edge[1]]:
                        # ignore same tag cases
                        if tag1 == tag2:
                            continue
                        edge_count[(tag1, tag2)] += 1 / nr_tag_pairs
                        # edge_count[(tag1, tag2)] += 1
        else:
            # not a v-structure
            edge_1 = (triple[1], triple[0])
            edge_2 = (triple[1], triple[2])
            both_edges = [edge_1, edge_2]
            # get tags that support the two type fork idea
            two_type_fork_tag_pairs = []
            for tag1 in tags[edge_1[0]]:
                for tag2 in tags[edge_1[1]]:
                    if tag1 == tag2:
                        continue
                    # tag must be in both ends of edges
                    if tag2 in tags[edge_2[1]]:
                        two_type_fork_tag_pairs.append((tag1, tag2))

            nr_tag_pairs = len(two_type_fork_tag_pairs)
            for tag1, tag2 in two_type_fork_tag_pairs:
                # using twice the weight, as the evidence is based on two edges
                edge_count[(tag1, tag2)] += 2 / nr_tag_pairs
                # edge_count[(tag1, tag2)] += 2

    # (here, typing orient using the typing edge with the biggest majority; this does not quite work for us as tags can contradict each other and the most confident tag pair might not be the most confident direction of an edge)

    # 4. greedily: go over each edge and direct the highest probability one

    # cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i --> j ,
    # cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
    # cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.
    graph = cg_1.G.graph

    # get directed and undirected edges
    directed_edges, undirected_edges = get_edges(graph)

    directed_edge_info = []

    # while edges are undirected
    while len(undirected_edges) > 0:
        # iterate over all undirected edges
        # remember edge with highest probability (first element is edge, second probability, third direction)
        # direction is True if edge is directed from first to second node
        best_edge_score_direction = find_best_to_direct(undirected_edges, tags, edge_count, min_samples, backward_only=False, var_tags=var_tags)
   
        if best_edge_score_direction[1] == 0.5 or best_edge_score_direction[1] < min_prob_threshold:
            break

        # after iteration, direct edge with highest score
        to_direct = best_edge_score_direction[0]

        if best_edge_score_direction[2]:
            cg_1.G.graph[to_direct[1], to_direct[0]] = 1
        else:    
            cg_1.G.graph[to_direct[0], to_direct[1]] = 1

        direction_valid = True
        if cg_1.G.exists_directed_cycle():
            # invert edge
            cg_1.G.graph[to_direct[0], to_direct[1]] *= -1
            cg_1.G.graph[to_direct[1], to_direct[0]] *= -1
            direction_valid = False
            print(f"Cycle detected, inverting edge")

        directed_edge_info.append((best_edge_score_direction[0], best_edge_score_direction[1], best_edge_score_direction[2], direction_valid, best_edge_score_direction[3], False))

        # get undirected edges  # TODO could do something more efficient and only update what changed but this one works as well
        directed_edges, undirected_edges = get_edges(cg_1.G.graph)

    # one final meeks
    cg = Meek.meek(cg_1, background_knowledge=background_knowledge)

    end = time.time()

    cg.PC_elapsed = end - start

    return cg, directed_edge_info


def mvpc_alg(
    data: ndarray,
    node_names: List[str] | None,
    alpha: float,
    indep_test: str,
    correction_name: str,
    stable: bool,
    uc_rule: int,
    uc_priority: int,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    **kwargs,
) -> CausalGraph:
    """
    Perform missing value Peter-Clark (PC) algorithm for causal discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)
    alpha :  float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : str, name of the test-wise deletion independence test being used
            ["mv_fisherz", "mv_g_sq"]
            - mv_fisherz: Fisher's Z conditional independence test
            - mv_g_sq: G-squared conditional independence test (TODO: under development)
    correction_name : correction_name: name of the missingness correction
            [MV_Crtn_Fisher_Z, MV_Crtn_G_sq, MV_DRW_Fisher_Z, MV_DRW_G_sq]
            - "MV_Crtn_Fisher_Z": Permutation based correction method
            - "MV_Crtn_G_sq": G-squared conditional independence test (TODO: under development)
            - "MV_DRW_Fisher_Z": density ratio weighting based correction method (TODO: under development)
            - "MV_DRW_G_sq": G-squared conditional independence test (TODO: under development)
    stable : run stabilized skeleton discovery if True (default = True)
    uc_rule : how unshielded colliders are oriented
           0: run uc_sepset
           1: run maxP
           2: run definiteMaxP
    uc_priority : rule of resolving conflicts between unshielded colliders
           -1: whatever is default in uc_rule
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge: background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    start = time.time()
    indep_test = CIT(data, indep_test, **kwargs)
    ## Step 1: detect the direct causes of missingness indicators
    prt_m = get_parent_missingness_pairs(data, alpha, indep_test, stable)
    # print('Finish detecting the parents of missingness indicators.  ')

    ## Step 2:
    ## a) Run PC algorithm with the 1st step skeleton;
    cg_pre = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, stable,
                                                  background_knowledge=background_knowledge,
                                                  verbose=verbose, show_progress=show_progress, node_names=node_names)
    if background_knowledge is not None:
        orient_by_background_knowledge(cg_pre, background_knowledge)

    cg_pre.to_nx_skeleton()
    # print('Finish skeleton search with test-wise deletion.')

    ## b) Correction of the extra edges
    cg_corr = skeleton_correction(data, alpha, correction_name, cg_pre, prt_m, stable)
    # print('Finish missingness correction.')

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_corr, background_knowledge)

    ## Step 3: Orient the edges
    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = UCSepset.uc_sepset(cg_corr, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.uc_sepset(cg_corr, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = UCSepset.maxp(cg_corr, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.maxp(cg_corr, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = UCSepset.definite_maxp(cg_corr, alpha, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.definite_maxp(cg_corr, alpha, background_knowledge=background_knowledge)
        cg_before = Meek.definite_meek(cg_2, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_before, background_knowledge=background_knowledge)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")
    end = time.time()

    cg.PC_elapsed = end - start

    return cg


#######################################################################################################################
## *********** Functions for Step 1 ***********
def get_parent_missingness_pairs(data: ndarray, alpha: float, indep_test, stable: bool = True) -> Dict[str, list]:
    """
    Detect the parents of missingness indicators
    If a missingness indicator has no parent, it will not be included in the result
    :param data: data set (numpy ndarray)
    :param alpha: desired significance level in (0, 1) (float)
    :param indep_test: name of the test-wise deletion independence test being used
        - "MV_Fisher_Z": Fisher's Z conditional independence test
        - "MV_G_sq": G-squared conditional independence test (TODO: under development)
    :param stable: run stabilized skeleton discovery if True (default = True)
    :return:
    cg: a CausalGraph object
    """
    parent_missingness_pairs = {'prt': [], 'm': []}

    ## Get the index of missingness indicators
    missingness_index = get_missingness_index(data)

    ## Get the index of parents of missingness indicators
    # If the missingness indicator has no parent, then it will not be collected in prt_m
    for missingness_i in missingness_index:
        parent_of_missingness_i = detect_parent(missingness_i, data, alpha, indep_test, stable)
        if not isempty(parent_of_missingness_i):
            parent_missingness_pairs['prt'].append(parent_of_missingness_i)
            parent_missingness_pairs['m'].append(missingness_i)
    return parent_missingness_pairs


def isempty(prt_r) -> bool:
    """Test whether the parent of a missingness indicator is empty"""
    return len(prt_r) == 0


def get_missingness_index(data: ndarray) -> List[int]:
    """Detect the parents of missingness indicators
    :param data: data set (numpy ndarray)
    :return:
    missingness_index: list, the index of missingness indicators
    """

    missingness_index = []
    _, ncol = np.shape(data)
    for i in range(ncol):
        if np.isnan(data[:, i]).any():
            missingness_index.append(i)
    return missingness_index


def detect_parent(r: int, data_: ndarray, alpha: float, indep_test, stable: bool = True) -> ndarray:
    """Detect the parents of a missingness indicator
    :param r: the missingness indicator
    :param data_: data set (numpy ndarray)
    :param alpha: desired significance level in (0, 1) (float)
    :param indep_test: name of the test-wise deletion independence test being used
        - "MV_Fisher_Z": Fisher's Z conditional independence test
        - "MV_G_sq": G-squared conditional independence test (TODO: under development)
    :param stable: run stabilized skeleton discovery if True (default = True)
    : return:
    prt: parent of the missingness indicator, r
    """
    ## TODO: in the test-wise deletion CI test, if test between a binary and a continuous variable,
    #  there can be the case where the binary variable only take one value after deletion.
    #  It is because the assumption is violated.

    ## *********** Adaptation 0 ***********
    # For avoid changing the original data
    data = data_.copy()
    ## *********** End ***********

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    ## *********** Adaptation 1 ***********
    # data
    ## Replace the variable r with its missingness indicator
    ## If r is not a missingness indicator, return [].
    data[:, r] = np.isnan(data[:, r]).astype(float)  # True is missing; false is not missing
    if sum(data[:, r]) == 0 or sum(data[:, r]) == len(data[:, r]):
        return np.empty(0)
    ## *********** End ***********

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var)
    cg.set_ind_test(CIT(data, indep_test.method))

    node_ids = range(no_of_var)
    pair_of_variables = list(permutations(node_ids, 2))

    depth = -1
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        for (x, y) in pair_of_variables:

            ## *********** Adaptation 2 ***********
            # the skeleton search
            ## Only test which variable is the neighbor of r
            if x != r:
                continue
            ## *********** End ***********

            Neigh_x = cg.neighbors(x)
            if y not in Neigh_x:
                continue
            else:
                Neigh_x = np.delete(Neigh_x, np.where(Neigh_x == y))

            if len(Neigh_x) >= depth:
                for S in combinations(Neigh_x, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if not stable:  # Unstable: Remove x---y right away
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                        else:  # Stable: x---y will be removed only
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            Helper.append_value(cg.sepset, x, y, S)
                            Helper.append_value(cg.sepset, y, x, S)
                        break

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    ## *********** Adaptation 3 ***********
    ## extract the parent of r from the graph
    cg.to_nx_skeleton()
    cg_skel_adj = nx.to_numpy_array(cg.nx_skel).astype(int)
    prt = get_parent(r, cg_skel_adj)
    ## *********** End ***********

    return prt


def get_parent(r: int, cg_skel_adj: ndarray) -> ndarray:
    """Get the neighbors of missingness indicators which are the parents
    :param r: the missingness indicator index
    :param cg_skel_adj: adjacency matrix of a causal skeleton
    :return:
    prt: list, parents of the missingness indicator r
    """
    num_var = len(cg_skel_adj[0, :])
    indx = np.array([i for i in range(num_var)])
    prt = indx[cg_skel_adj[r, :] == 1]
    return prt


## *********** END ***********
#######################################################################################################################

def skeleton_correction(data: ndarray, alpha: float, test_with_correction_name: str, init_cg: CausalGraph, prt_m: dict,
                        stable: bool = True) -> CausalGraph:
    """Perform skeleton discovery
    :param data: data set (numpy ndarray)
    :param alpha: desired significance level in (0, 1) (float)
    :param test_with_correction_name: name of the independence test being used
           - "MV_Crtn_Fisher_Z": Fisher's Z conditional independence test
           - "MV_Crtn_G_sq": G-squared conditional independence test
    :param stable: run stabilized skeleton discovery if True (default = True)
    :return:
    cg: a CausalGraph object
    """

    assert type(data) == np.ndarray
    assert 0 < alpha < 1
    assert test_with_correction_name in ["MV_Crtn_Fisher_Z", "MV_Crtn_G_sq"]

    ## *********** Adaption 1 ***********
    no_of_var = data.shape[1]

    ## Initialize the graph with the result of test-wise deletion skeletion search
    cg = init_cg

    if test_with_correction_name in ["MV_Crtn_Fisher_Z", "MV_Crtn_G_sq"]:
        cg.set_ind_test(CIT(data, "mc_fisherz"))
    # No need of the correlation matrix if using test-wise deletion test
    cg.prt_m = prt_m
    ## *********** Adaption 1 ***********

    node_ids = range(no_of_var)
    pair_of_variables = list(permutations(node_ids, 2))

    depth = -1
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        for (x, y) in pair_of_variables:
            Neigh_x = cg.neighbors(x)
            if y not in Neigh_x:
                continue
            else:
                Neigh_x = np.delete(Neigh_x, np.where(Neigh_x == y))

            if len(Neigh_x) >= depth:
                for S in combinations(Neigh_x, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if not stable:  # Unstable: Remove x---y right away
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                        else:  # Stable: x---y will be removed only
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            Helper.append_value(cg.sepset, x, y, S)
                            Helper.append_value(cg.sepset, y, x, S)
                        break

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    return cg


#######################################################################################################################

# *********** Evaluation util ***********

def get_adjacancy_matrix(g: CausalGraph) -> ndarray:
    return nx.to_numpy_array(g.nx_graph).astype(int)


def matrix_diff(cg1: CausalGraph, cg2: CausalGraph) -> (float, List[Tuple[int, int]]):
    adj1 = get_adjacancy_matrix(cg1)
    adj2 = get_adjacancy_matrix(cg2)
    count = 0
    diff_ls = []
    for i in range(len(adj1[:, ])):
        for j in range(len(adj2[:, ])):
            if adj1[i, j] != adj2[i, j]:
                diff_ls.append((i, j))
                count += 1
    return count / 2, diff_ls