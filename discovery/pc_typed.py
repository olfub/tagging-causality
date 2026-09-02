# copied and adapted from https://github.com/ServiceNow/typed-dag

from collections import Counter, defaultdict
from itertools import combinations, permutations, product
from random import shuffle

import networkx as nx
import numpy as np
from discovery.pc_typed_graph_utils import _has_any_edge, _has_both_edges, _orient, type_of

from typing import Tuple
from causaldag import DAG, PDAG

def tpc_naive(skeleton, separating_sets):
    out = pc_meek_rules(orient_forks_naive(skeleton=skeleton, sep_sets=separating_sets))
    return nx.adjacency_matrix(out).todense()


def tpc_majority_top1(skeleton, separating_sets):
    out = pc_meek_rules(orient_forks_majority_top1(skeleton=skeleton, sep_sets=separating_sets))
    return nx.adjacency_matrix(out).todense()

def pc_meek_rules(dag):
    """
    Step 3: Meek rules portion of the PC algorithm

    """
    node_ids = dag.nodes()

    # For all the combination of nodes i and j, apply the following
    # rules.
    old_dag = dag.copy()
    while True:
        for (i, j) in permutations(node_ids, 2):
            # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
            # such that k and j are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Look all the predecessors of i.
                for k in dag.predecessors(i):
                    # Skip if there is an arrow i->k.
                    if dag.has_edge(i, k):
                        continue
                    # Skip if k and j are adjacent.
                    if _has_any_edge(dag, k, j):
                        continue
                    # Make i-j into i->j
                    # logging.debug("R1: remove edge (%s, %s)" % (j, i))
                    _orient(dag, i, j)
                    break

            # Rule 2: Orient i-j into i->j whenever there is a chain
            # i->k->j.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where k is i->k.
                succs_i = set()
                for k in dag.successors(i):
                    if not dag.has_edge(k, i):
                        succs_i.add(k)
                # Find nodes j where j is k->j.
                preds_j = set()
                for k in dag.predecessors(j):
                    if not dag.has_edge(j, k):
                        preds_j.add(k)
                # Check if there is any node k where i->k->j.
                if len(succs_i & preds_j) > 0:
                    # Make i-j into i->j
                    # logging.debug("R2: remove edge (%s, %s)" % (j, i))
                    _orient(dag, i, j)

            # Rule 3: Orient i-j into i->j whenever there are two chains
            # i-k->j and i-l->j such that k and l are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where i-k.
                adj_i = set()
                for k in dag.successors(i):
                    if dag.has_edge(k, i):
                        adj_i.add(k)
                # For all the pairs of nodes in adj_i,
                for (k, l) in combinations(adj_i, 2):
                    # Skip if k and l are adjacent.
                    if _has_any_edge(dag, k, l):
                        continue
                    # Skip if not k->j.
                    if dag.has_edge(j, k) or (not dag.has_edge(k, j)):
                        continue
                    # Skip if not l->j.
                    if dag.has_edge(j, l) or (not dag.has_edge(l, j)):
                        continue
                    # Make i-j into i->j.
                    # logging.debug("R3: remove edge (%s, %s)" % (j, i))
                    _orient(dag, i, j)
                    break

            # Rule 4: Orient i-j into i->j whenever there are two chains
            # i-k->l and k->l->j such that k and j are nonadjacent.
            # TODO: validate me
            if _has_both_edges(dag, i, j):
                # Find nodes k where i-k.
                adj_i = set()
                for k in dag.successors(i):
                    if dag.has_edge(k, i):
                        adj_i.add(k)

                # Find nodes l where l -> j
                preds_j = set()
                for l in dag.predecessors(j):
                    if not dag.has_edge(j, l):
                        preds_j.add(l)

                # Find nodes where k -> l
                for k in adj_i:
                    for l in preds_j:
                        if dag.has_edge(k, l) and not dag.has_edge(l, k):
                            _orient(dag, i, j)
                            break

        if nx.is_isomorphic(dag, old_dag):
            break
        old_dag = dag.copy()

    return dag


def orient_forks_naive(skeleton, sep_sets):
    """
    Orient immoralities and two-type forks

    Strategy: naive -- orient as first encountered

    """
    dag = skeleton.to_directed()
    node_ids = skeleton.nodes()

    # Orient all immoralities and two-type forks
    # TODO: DEBUG using shuffling to test hypothesis
    combos = list(combinations(node_ids, 2))
    shuffle(combos)
    for (i, j) in combos:
        adj_i = set(dag.successors(i))
        adj_j = set(dag.successors(j))

        # If j is a direct child of i
        if j in adj_i:
            continue

        # If i is a direct child of j
        if i in adj_j:
            continue

        # If i and j are directly connected, continue.
        if sep_sets[i][j] is None:
            continue

        common_k = adj_i & adj_j  # Common direct children of i and j
        for k in common_k:
            # Case: we have an immorality i -> k <- j
            if k not in sep_sets[i][j] and k in dag.successors(i) and k in dag.successors(j):
                # XXX: had to add the last two conditions in case k is no longer a child due to t-edge orientation
                # logging.debug(
                #     f"S: orient immorality {i} (t{type_of(dag, i)}) -> {k} (t{type_of(dag, k)}) <- {j} (t{type_of(dag, j)})"
                # )
                _orient(dag, i, k)
                _orient(dag, j, k)

            # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j
            elif (
                type_of(dag, i) == type_of(dag, j)
                and type_of(dag, i) != type_of(dag, k)
                and _has_both_edges(dag, i, k)
                and _has_both_edges(dag, j, k)
            ):
                # logging.debug(
                #     f"S: orient two-type fork {i} (t{type_of(dag, i)}) <- {k} (t{type_of(dag, k)}) -> {j} (t{type_of(dag, j)})"
                # )
                _orient(dag, k, i)  # No need to orient k -> j. Will be done in this call since i,j have the same type.

    return dag


def orient_forks_majority_top1(skeleton, sep_sets):
    """
    Orient immoralities and two-type forks

    Strategy: majority -- orient using the most frequent orientation
    Particularity: Find the t-edge with most evidence, orient, repeat evidence collection.

    """
    dag = skeleton.to_directed()
    node_ids = skeleton.nodes()
    n_types = len(np.unique([type_of(dag, n) for n in dag.nodes()]))

    oriented_tedge = True
    while oriented_tedge:

        # Accumulator for evidence of t-edge orientation
        # We will count how often we see the t-edge in each direction and choose the most frequent one.
        tedge_evidence = np.zeros((n_types, n_types))
        oriented_tedge = False

        # Some immoralities will contain edges between variables of the same type. These will not be
        # automatically oriented when we decide on the t-edge orientations. To make sure that we orient
        # them correctly, we maintain a list of conditional orientations, i.e., how should an intra-type
        # edge be oriented if we make a specific orientation decision for the t-edges.
        conditional_orientations = defaultdict(list)

        # Step 1: Gather evidence of orientation for all immoralities and two-type forks that involve more than one type
        for (i, j) in combinations(node_ids, 2):
            adj_i = set(dag.successors(i))
            adj_j = set(dag.successors(j))

            # If j is a direct child of i, i is a direct child of j, or ij are directly connected
            if j in adj_i or i in adj_j or sep_sets[i][j] is None:
                continue

            for k in adj_i & adj_j:  # Common direct children of i and j
                # Case: we have an immorality i -> k <- j
                if k not in sep_sets[i][j]:
                    # Check if already oriented
                    # if not _has_both_edges(dag, i, k) or not _has_both_edges(dag, j, k):
                    #     continue
                    if not _has_both_edges(dag, i, k) and not _has_both_edges(dag, j, k):
                        # Fully oriented
                        continue

                    # Ensure that we don't have only one type. We will orient these later.
                    if type_of(dag, i) == type_of(dag, j) == type_of(dag, k):
                        continue

                    # logging.debug(
                    #     f"Step 1: evidence of orientation {i} (t{type_of(dag, i)}) -> {k} (t{type_of(dag, k)}) <- {j} (t{type_of(dag, j)})"
                    # )
                    # Increment t-edge orientation evidence
                    tedge_evidence[type_of(dag, i), type_of(dag, k)] += 1
                    tedge_evidence[type_of(dag, j), type_of(dag, k)] += 1

                    # Determine conditional orientations
                    conditional_orientations[(type_of(dag, j), type_of(dag, k))].append((i, k))
                    conditional_orientations[(type_of(dag, i), type_of(dag, k))].append((j, k))

                # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j
                elif type_of(dag, i) == type_of(dag, j) and type_of(dag, i) != type_of(dag, k):
                    # Check if already oriented
                    if not _has_both_edges(dag, i, k) or not _has_both_edges(dag, j, k):
                        continue

                    # logging.debug(
                    #     f"Step 1: evidence of orientation {i} (t{type_of(dag, i)}) <- {k} (t{type_of(dag, k)}) -> {j} (t{type_of(dag, j)})"
                    # )
                    # Count evidence only once per t-edge
                    tedge_evidence[type_of(dag, k), type_of(dag, i)] += 2

        # Step 2: Orient t-edges based on evidence
        np.fill_diagonal(tedge_evidence, 0)
        ti, tj = np.unravel_index(tedge_evidence.argmax(), tedge_evidence.shape)
        if np.isclose(tedge_evidence[ti, tj], 0):
            continue

        # Orient!
        # print("Evidence", tedge_evidence[ti, tj])
        # print(conditional_orientations)
        oriented_tedge = True
        first_ti = [n for n in dag.nodes() if type_of(dag, n) == ti][0]
        first_tj = [n for n in dag.nodes() if type_of(dag, n) == tj][0]
        # logging.debug(
        #     f"Step 2: orienting t-edge according to max evidence. t{ti} -> t{tj} ({tedge_evidence[ti, tj]}) vs t{ti} <- t{tj} ({tedge_evidence[tj, ti]})"
        # )
        _orient(dag, first_ti, first_tj)
        cond = Counter(conditional_orientations[ti, tj])
        for (n1, n2), count in cond.items():
            # logging.debug(f"... conditional orientation {n1}->{n2} (count: {count}).")
            if (n2, n1) in cond and cond[n2, n1] > count:
                # logging.debug(
                #     f"Skipping this one. Will orient its counterpart ({n2}, {n1}) since it's more frequent: {cond[n2, n1]}."
                # )
                pass
            else:
                _orient(dag, n1, n2)
    # logging.debug("Steps 1-2 completed. Moving to single-type forks.")

    # Step 3: Orient remaining immoralities (all variables of the same type)
    for (i, j) in combinations(node_ids, 2):
        adj_i = set(dag.successors(i))
        adj_j = set(dag.successors(j))

        # If j is a direct child of i, i is a direct child of j, ij are directly connected
        if j in adj_i or i in adj_j or sep_sets[i][j] is None:
            continue

        for k in adj_i & adj_j:  # Common direct children of i and j
            # Case: we have an immorality i -> k <- j
            if k not in sep_sets[i][j]:
                # Only single-type immoralities
                if not (type_of(dag, i) == type_of(dag, j) == type_of(dag, k)):
                    continue
                # logging.debug(
                #     f"Step 3: orient immorality {i} (t{type_of(dag, i)}) -> {k} (t{type_of(dag, k)}) <- {j} (t{type_of(dag, j)})"
                # )
                _orient(dag, i, k)
                _orient(dag, j, k)

    return dag

def is_acyclic(adjacency: np.ndarray) -> bool:
    """
    Check if adjacency matrix is acyclic
    :param adjacency: adjacency matrix
    :returns: True if acyclic
    """
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0:
            return False
    return True


def has_same_immoralities(g1: np.ndarray, g2: np.ndarray) -> bool:
    """
    Check if g1 and g2 have the same immoralities
    :param g1: adjacency matrix of a graph (can be only partially directed)
    :param g2: adjacency matrix of a graph (can be only partially directed)
    :returns: True if they have the same immoralities
    """
    for node in range(g1.shape[0]):
        for par1 in range(g1.shape[0]):
            for par2 in range(g1.shape[0]):
                # check if have parents that are not married
                if par1 != par2:
                    if (
                        g1[par1, node] == 1
                        and g1[par2, node] == 1
                        and g1[node, par1] == 0
                        and g1[node, par2] == 0
                        and g1[par1, par2] == 0
                        and g1[par2, par1] == 0
                    ):
                        if not (
                            g2[par1, node] == 1
                            and g2[par2, node]
                            and g2[node, par1] == 0
                            and g2[node, par2] == 0
                            and g2[par1, par2] == 0
                            and g2[par2, par1] == 0
                        ):
                            return False

    for node in range(g2.shape[0]):
        for par1 in range(g2.shape[0]):
            for par2 in range(g2.shape[0]):
                # check if parents are not married
                if par1 != par2:
                    if (
                        g2[par1, node] == 1
                        and g2[par2, node] == 1
                        and g2[node, par1] == 0
                        and g2[node, par2] == 0
                        and g2[par1, par2] == 0
                        and g2[par2, par1] == 0
                    ):
                        if not (
                            g1[par1, node] == 1
                            and g1[par2, node]
                            and g1[node, par1] == 0
                            and g1[node, par2] == 0
                            and g1[par1, par2] == 0
                            and g1[par2, par1] == 0
                        ):
                            return False

    return True


class EmptySetException(Exception):
    pass

def _update_tedge_orientation(G, type_g, types):
    """
    Detects which t-edges are oriented and unoriented and updates the type compatibility graph

    """
    type_g = np.copy(type_g)

    for a, b in permutations(range(G.shape[0]), 2):
        # XXX: No need to consider the same-type case, since the type matrix starts at identity.
        if types[a] == types[b]:
            continue
        # Detect unoriented t-edges
        if G[a, b] == 1 and G[b, a] == 1 and not (type_g[types[a], types[b]] + type_g[types[b], types[a]] == 1):
            type_g[types[a], types[b]] = 1
            type_g[types[b], types[a]] = 1
        # Detect oriented t-edges
        if G[a, b] == 1 and G[b, a] == 0:
            type_g[types[a], types[b]] = 1
            type_g[types[b], types[a]] = 0

    return type_g


def _orient_tedges(G, type_g, types):
    """
    Ensures that edges that belong to oriented t-edges are consistently oriented.

    Note: will not change the orientation of edges that are already oriented, even if they clash with the direction
          of the t-edge. This can happen if the CPDAG was not type consistant at the start of t-Meek.

    """
    G = np.copy(G)
    for a, b in permutations(range(G.shape[0]), 2):
        if type_g[types[a], types[b]] == 1 and type_g[types[b], types[a]] == 0 and G[a, b] == 1 and G[b, a] == 1:
            G[a, b] = 1
            G[b, a] = 0
    return G


def typed_meek(cpdag: np.ndarray, types: list, iter_max: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the Meek algorithm with the type consistency as described in Section 5 (from the CPDAG).

    :param cpdag: adjacency matrix of the CPDAG
    :param types: list of type of each node
    :param iter_max: The maximum number of iterations. If reached, an exception will be raised.
    """
    n_nodes = cpdag.shape[0]
    types = np.asarray(types)
    n_types = len(np.unique(types))

    G = np.copy(cpdag)
    type_g = np.eye(n_types)  # Identity matrix to allow intra-type edges

    # repeat until the graph is not changed by the algorithm
    # or too high number of iteration
    previous_G = np.copy(G)
    i = 0
    while True and i < iter_max:
        """
        Each iteration is broken down into three stages:
        1) Update t-edge orientations based on the CPDAG
        2) Orient all edges that are part of the same t-edge consistently (if the t-edge is oriented)
        3) Apply the Meek rules (only one per iteration) to orient the remaining edges.

        Note: Edges are oriented one-by-one in step 3, but these orientations will be propagated to the whole
              t-edge once we return to step (1).

        """
        i += 1
        # Step 1: Determine the orientation of all t-edges based on t-edges (some may be unoriented)
        type_g = _update_tedge_orientation(G, type_g, types)

        # Step 2: Orient all edges of the same type in the same direction if their t-edge is oriented.
        # XXX: This will not change the orientation of oriented edges (i.e., if the CPDAG was not consistant)
        G = _orient_tedges(G, type_g, types)

        # Step 3: Apply Meek's rules (R1, R2, R3, R4) and the two-type fork rule (R5)
        for a, b, c in permutations(range(n_nodes), 3):
            # Orient any undirected edge a - b as a -> b if any of the following rules is satisfied:
            if G[a, b] != 1 or G[b, a] != 1:
                # Edge a - b is already oriented
                continue

            # R1: c -> a - b ==> a -> b
            if G[a, c] == 0 and G[c, a] == 1 and G[b, c] == 0 and G[c, b] == 0:
                G[b, a] = 0
            # R2: a -> c -> b and a - b ==> a -> b
            elif G[a, c] == 1 and G[c, a] == 0 and G[b, c] == 0 and G[c, b] == 1:
                G[b, a] = 0
            # R5: b - a - c and t(c) = t(b) ==> a -> b and a -> c (two-type fork)
            elif (
                G[a, c] == 1
                and G[c, a] == 1
                and G[b, c] == 0
                and G[c, b] == 0
                and types[b] == types[c]
                and types[b] != types[a]  # Make sure there are at least two types
            ):
                G[b, a] = 0
                G[c, a] = 0
            else:

                for d in range(n_nodes):
                    if d != a and d != b and d != c:
                        # R3: a - c -> b and a - d -> b, c -/- d ==> a -> b, and a - b
                        if (
                            G[a, c] == 1
                            and G[c, a] == 1
                            and G[b, c] == 0
                            and G[c, b] == 1
                            and G[a, d] == 1
                            and G[d, a] == 1
                            and G[b, d] == 0
                            and G[d, b] == 1
                            and G[c, d] == 0
                            and G[d, c] == 0
                        ):
                            G[b, a] = 0
                        # R4: a - d -> c -> b and a - - c ==> a -> b
                        elif (
                            G[a, d] == 1
                            and G[d, a] == 1
                            and G[c, d] == 0
                            and G[d, c] == 1
                            and G[b, c] == 0
                            and G[c, b] == 1
                            and (G[a, c] == 1 or G[c, a] == 1)
                        ):
                            G[b, a] = 0

        if (previous_G == G).all():
            break
        if i >= iter_max:
            raise Exception(f"Typed Meek is stucked. More than {iter_max} iterations.")

        previous_G = np.copy(G)

    return G, type_g


def tmec_enumeration(cpdag: np.ndarray, types: list, type_g: np.ndarray) -> list:
    """
    Finds all the possible DAGs represented by a t-CPDAG
    :param cpdag: A PDAG that does not violate type consistency
    :param types: list of types of each node
    :param type_g: adjacency matrix of the graph over types
    :returns: the list of possible DAGs
    """
    n_nodes = cpdag.shape[0]
    n_types = len(np.unique(types))

    type_g = np.copy(type_g)

    # Find every unoriented t-edge
    unoriented_tedges = []
    for i, j in combinations(range(n_types), 2):
        if type_g[i, j] == 1 and type_g[j, i] == 1:
            unoriented_tedges.append([i, j])

    # Enumerate every possible orientation for each unoriented t-edge
    dag_list = []
    for orientation in product([0, 1], repeat=len(unoriented_tedges)):
        t_cpdag = np.copy(cpdag)
        oriented_type_g = np.copy(type_g)

        # Orient each undirected t-edge with a given orientation
        for i, edge_orientation in enumerate(orientation):
            oriented_type_g[unoriented_tedges[i][0], unoriented_tedges[i][1]] = edge_orientation
            oriented_type_g[unoriented_tedges[i][1], unoriented_tedges[i][0]] = 1 - edge_orientation

        # Orient all unoriented inter-type edges
        for i, j in permutations(range(n_nodes), 2):
            if types[i] == types[j] or t_cpdag[i, j] + t_cpdag[j, i] != 2:
                # Either an intra-type edge or an inter-type that is already oriented. Don't touch.
                continue

            # Make sure the t-edge between these variables exists (i.e., types are connected)
            assert (
                oriented_type_g[types[i], types[j]] + oriented_type_g[types[j], types[i]] == 1
            ), "Found connected nodes with no t-edge."

            # Orient according to t-edge orientation
            if oriented_type_g[types[i], types[j]] == 1:
                t_cpdag[j, i] = 0
            else:
                t_cpdag[i, j] = 0

        # Enumerate all DAGs that can be generated from the current t-CPDAG
        # Edges that remain to be oriented are intra-type.
        for d in PDAG.from_amat(t_cpdag).all_dags():
            d = DAG(nodes=np.arange(t_cpdag.shape[0]), arcs=d).to_amat()[0]
            # Add DAG only if it is acyclic,
            # it does not have extra v-structures,
            # and its skeleton is identical
            # XXX: this last condition is required since sometimes t-edge orientations lead to impossible
            #      t-essential graphs and causaldag returns incorrect graphs with missing edges.
            if (
                is_acyclic(d)
                and has_same_immoralities(d, cpdag)
                and PDAG.from_amat(t_cpdag).skeleton == DAG.from_amat(d).skeleton
            ):
                dag_list.append(d)

    if len(dag_list) == 0:
        raise EmptySetException(
            "Error: t-MEC enumeration returned no valid t-DAG. CPDAG probably violates type consistency or constains a cycle."
        )

    return dag_list