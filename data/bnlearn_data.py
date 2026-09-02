# import warnings
# # throw every warning as an error
# warnings.filterwarnings("error")
import pickle
import random

import bnlearn as bn
import numpy as np
import torch


def get_data(identifier, nr_samples, seed):
    if identifier in ["child", "earthquake", "insurance", "survey", "asia", "cancer", "alarm", "hepar2", "hailfinder", "win95pts"]:
        with open(f"data/bnlearn_files/{seed}/{identifier}_data_{nr_samples}.pkl", "rb") as f:
            data = pickle.load(f)
        with open(f"data/bnlearn_files/{seed}/{identifier}_edges_{nr_samples}.pkl", "rb") as f:
            edges = pickle.load(f)
    else:
        raise RuntimeError(f"For experiments currently, all data should be loaded from the pre-generated pickle files. Dataset {identifier} not found.")
        if identifier in ["cancer", "child", "earthquake", "insurance", "survey", "hailfinder", "win95pts", "hepar2"]:
            file_path = f"data/bnlearn_files/{identifier}.bif"
            graph = bn.import_DAG(file_path)
            data = bn.sampling(graph, n=nr_samples)
        elif identifier in ["asia", "alarm"]:
            graph = bn.import_DAG(identifier)
            data = bn.import_example(identifier, n=nr_samples)
            if data is None:
                # often, loading the second time works... I know, it's weird
                data = bn.import_example(identifier)
        else:
            raise ValueError(f"Unknown dataset identifier: {identifier}")
        adjmat = graph["adjmat"]
        edges = []
        for var in data.columns:
            for var2 in data.columns:
                if adjmat.loc[var, var2] == 1:
                    edges.append((var, var2))

    variables = list(data.columns)
    var_labels = list(data.columns)
    tags = None
    edges = edges
    positions = None
    data = data.to_numpy()
    return variables, var_labels, tags, edges, positions, data

def make_deterministic(seed=0):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    for i in range(10):
        make_deterministic(i)
        nr_samples = 10000
        identifiers = ["child", "earthquake", "insurance", "survey", "asia", "cancer", "alarm", "hepar2", "win95pts", "hailfinder"]
        for identifier in identifiers:
            if identifier in ["cancer", "child", "earthquake", "insurance", "survey", "hepar2", "win95pts", "hailfinder"]:
                file_path = f"data/bnlearn_files/{identifier}.bif"
                graph = bn.import_DAG(file_path)
                data = bn.sampling(graph, n=nr_samples)
            elif identifier in ["asia", "alarm"]:
                graph = bn.import_DAG(identifier)
                data = bn.import_example(identifier, n=nr_samples)
                if data is None:
                    # often, loading the second time works... I know, it's weird
                    data = bn.import_example(identifier)
            else:
                raise ValueError(f"Unknown dataset identifier: {identifier}")
            adjmat = graph["adjmat"]
            edges = []
            for var in data.columns:
                for var2 in data.columns:
                    if adjmat.loc[var, var2] == 1:
                        edges.append((var, var2))
            data.to_pickle(f"data/bnlearn_files/{i}/{identifier}_data_{nr_samples}.pkl")
            with open(f"data/bnlearn_files/{i}/{identifier}_edges_{nr_samples}.pkl", "wb") as f:
                pickle.dump(edges, f)


if __name__ == "__main__":
    main()
