import networkx as nx
import numpy as np


class GenerateError(RuntimeError):
    pass


class ANM:
    def __init__(self, seed=None):
        self.node_biases = None
        self.node_noise = None
        self.node_names = None
        self.coeff_bounds = None
        self.neg_coeffs = None
        self.graph: nx.DiGraph = None
        self.rng = np.random.default_rng(seed)

    def _random_weights(self, size):
        # create weights within the coeff_bounds, possibly including negative values
        # by using a self.coeff_bounds[0] > 0, we can avoid small absolute values
        weights = self.rng.uniform(
            low=self.coeff_bounds[0], high=self.coeff_bounds[1], size=size
        )
        if self.neg_coeffs:
            factors = self.rng.choice([1, -1], size=size)
        weights *= factors
        return weights

    def generate_new_dag(
        self,
        nr_nodes,
        nr_edges,
        coeff_bounds=(1, 10),
        neg_coeffs=True,
        bias_bounds=(-10, 10),
        noise_scale_bounds=(0.1, 1),
        graph=None,
    ):
        self.coeff_bounds = coeff_bounds
        self.neg_coeffs = neg_coeffs
        self.bias_bounds = bias_bounds

        if graph is not None:
            nodes, edges = graph
            assert nr_nodes == len(nodes)
            assert nr_edges == len(edges)
            adj = np.zeros((len(nodes), len(nodes)))
            weights = self._random_weights(len(edges))
            for edge, weight in zip(edges, weights):
                adj[nodes.index(edge[0]), nodes.index(edge[1])] = weight
            self.graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
            self.node_names = nodes
        else:
            # adjacency matrix
            adj = self._random_weights((nr_nodes, nr_nodes))
            adj = np.tril(adj, -1)
            edge_indices = np.nonzero(adj)
            nr_current_edges = len(edge_indices[0])
            if nr_edges > nr_current_edges:
                GenerateError(
                    f"Too many edges specified, can not be satisfied (want {nr_edges} but maximum DAG has {nr_current_edges})"
                )
            nr_edges_to_rm = nr_current_edges - nr_edges
            edges_to_rm = self.rng.choice(
                np.arange(nr_current_edges), nr_edges_to_rm, replace=False
            )
            adj[edge_indices[0][edges_to_rm], edge_indices[1][edges_to_rm]] = 0
            self.graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
            self.node_names = [f"node_{i}" for i in range(nr_nodes)]

        # node bias (could also be the mean of the gaussian noise, but here coded as "bias")
        self.node_biases = self.rng.uniform(
            low=bias_bounds[0], high=bias_bounds[1], size=nr_nodes
        )
        self.node_noise = np.zeros(
            (nr_nodes, 2)
        )  # for each node: first index is loc, second is scale
        # keep loc at 0 but set random uniform scale
        self.node_noise[:, 1] = self.rng.uniform(
            low=noise_scale_bounds[0], high=noise_scale_bounds[1], size=nr_nodes
        )

    def sample(self, number):
        # make a specified number of samples
        values = np.zeros((number, len(self.graph.nodes)))
        node_order = list(nx.topological_sort(self.graph))
        for node_id in node_order:
            in_edges = self.graph.in_edges(node_id)
            for edge in in_edges:
                parent = edge[0]
                assert edge[1] == node_id
                weight = nx.adjacency_matrix(self.graph)[parent, node_id]
                values[:, node_id] += values[:, parent] * weight
            self.rng.normal(
                self.node_noise[node_id, 0], self.node_noise[node_id, 1], size=number
            )
            values[:, node_id] += (
                self.rng.normal(
                    self.node_noise[node_id, 0],
                    self.node_noise[node_id, 1],
                    size=number,
                )
                + self.node_biases[node_id]
            )
        return values


# test_anm = ANM(seed=42)
# test_anm.generate_new_dag(5, 5)
# print(test_anm.sample(10))
