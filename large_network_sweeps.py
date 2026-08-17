import contextlib
import io

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from core.truth_table import TruthTableDataset
from utils.topology import (
    get_3_reachability,
    get_num_connected_components,
    get_num_cycles,
    get_p_sparse_DAG,
    get_p_sparse_DAG_random_ordering,
    get_random_overlapping_cycles_nets,
    get_reachable_no_cycles_mask,
    topology_to_motifs_vector,
)
from utils.training import train_model
from utils.truth_tables import get_random_TT_from_coin_flips, get_truth_table_by_idx


def _sample_ER(n, p):
    return nx.to_numpy_array(nx.erdos_renyi_graph(n, p, directed=True))


NETWORK_SAMPLERS = {
    "ER": _sample_ER,
    "SparseDAG": get_p_sparse_DAG,
    "SparseDAG_RandomOrdering": get_p_sparse_DAG_random_ordering,
    "ReachableNoCycles": get_reachable_no_cycles_mask,
    "cycles": get_random_overlapping_cycles_nets,
}

MODEL_PAPER_NAMES = {
    "ER": "Erdos-Renyi (ER)",
    "SparseDAG": "Input-expanding DAG",
    "SparseDAG_RandomOrdering": "Random DAG",
    "ReachableNoCycles": "Reachability without 3-cycles",
    "cycles": "Enriched 3-cycles",
}

FIG_5BC_CONFIG = {
    "n_vals": [3, 4, 5, 6, 7, 8, 9, 10],
    "p_vals": [0.1, 0.2, 0.5, 1.0],
    "nets_per_config": 100,
    "funcs_per_net": 100,
    "n_seeds": 10,
    "epochs": 500,
}

FIG_5E_6BC_CONFIG = {
    "func_dim": 10,
    "hidden_sizes": [10, 20, 30, 40, 50, 60, 80, 100],
    "p_vals": [0.05, 0.1, 0.15, 0.2],
    "network_models": ["ER", "SparseDAG", "SparseDAG_RandomOrdering",
                       "ReachableNoCycles", "cycles"],
    "n_functions": 100,
    "nets_per_config": 100,
    "n_seeds": 10,
    "epochs": 500,
}

FIG_6A_CONFIG = {
    "func_dim": 10,
    "hidden_sizes": list(range(10, 101)),
    "p_vals": sorted([round(0.05 + 0.01 * i, 3) for i in range(21)]
                     + [0.075, 0.125, 0.175, 0.225]),
    "network_models": ["ER", "SparseDAG", "SparseDAG_RandomOrdering", "ReachableNoCycles"],
    "n_functions": 100,
    "nets_per_config": 100,
    "n_seeds": 5,
    "epochs": 500,
    "extra_features": False,
}

FIG_6B_CONFIG = {
    "sizes": list(range(10, 101, 5)),
    "p_vals": [0.05, 0.1, 0.15, 0.2],
    "n_networks": 100,
}

DEMO_CONFIG = {
    "func_dim": 10,
    "hidden_sizes": [10, 20],
    "p_vals": [0.1],
    "network_models": ["ER", "SparseDAG", "SparseDAG_RandomOrdering",
                       "ReachableNoCycles", "cycles"],
    "n_functions": 2,
    "nets_per_config": 1,
    "n_seeds": 1,
    "epochs": 100,
}

DEMO_5BC_CONFIG = {
    "n_vals": [4, 6],
    "p_vals": [0.5],
    "nets_per_config": 2,
    "funcs_per_net": 2,
    "n_seeds": 1,
    "epochs": 100,
}


def sample_network(network_model, n, p):
    if network_model not in NETWORK_SAMPLERS:
        raise ValueError(f"Unknown network model: {network_model}")
    return NETWORK_SAMPLERS[network_model](n, p)


def sweep_size(config):
    return (len(config["hidden_sizes"])
            * len(config["p_vals"])
            * len(config["network_models"])
            * config["n_functions"]
            * config["nets_per_config"])


def train_one(func_dim, hidden_size, W, TT, n_seeds=5, epochs=500, lr=0.05,
              num_steps=3, batch_size=32, threshold=0.5, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TruthTableDataset(TT)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    mask = torch.tensor(W, dtype=torch.float32).to(device)
    accuracy, _ = train_model(n_seeds, func_dim, hidden_size, 1, num_steps, mask,
                              epochs, lr, threshold, device, dataloader, dataset)
    return accuracy


def _network_record(W, n, extra_features):
    record = {
        "num_edges": int(W.sum()),
        "density": W.sum() / (n ** 2 - n),
        "num_cycles": int(get_num_cycles(W)),
    }
    if extra_features:
        record["connected_components"] = get_num_connected_components(W)
        record["3_reachability"] = get_3_reachability(W)
        record["motif_vector"] = topology_to_motifs_vector(W, n)
    return record


def run_network_model_sweep(func_dim, hidden_sizes, p_vals, network_models,
                            nets_per_config, n_functions=None, functions=None,
                            n_seeds=5, epochs=500, lr=0.05, num_steps=3,
                            extra_features=True, device=None, progress=True):
    if functions is None:
        if n_functions is None:
            raise ValueError("Provide either `functions` or `n_functions`")
        functions = [get_random_TT_from_coin_flips(func_dim, 0.5) for _ in range(n_functions)]
    else:
        functions = [get_truth_table_by_idx(f, func_dim) if isinstance(f, int) else f
                     for f in functions]

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid = [(m, n, p) for m in network_models for n in hidden_sizes for p in p_vals]
    rows = []
    # train_model and get_reachable_no_cycles_mask print progress; the tqdm bar replaces it
    sink = io.StringIO()
    for network_model, hidden_size, p in tqdm(grid, disable=not progress, desc="sweep"):
        for net_idx in range(nets_per_config):
            with contextlib.redirect_stdout(sink):
                W = sample_network(network_model, hidden_size, p)
            record = _network_record(W, hidden_size, extra_features)
            for func_idx, TT in enumerate(functions):
                with contextlib.redirect_stdout(sink):
                    accuracy = train_one(func_dim, hidden_size, W, TT, n_seeds=n_seeds,
                                         epochs=epochs, lr=lr, num_steps=num_steps,
                                         device=device)
                rows.append({"func_idx": func_idx, "net_idx": net_idx,
                             "hidden_size": hidden_size, "p": p,
                             "network_model": network_model, "accuracy": accuracy,
                             **record})
    return pd.DataFrame(rows)


def run_small_network_sweep(n_vals, p_vals, nets_per_config, funcs_per_net=20,
                            network_model="ER", n_seeds=5, epochs=300, lr=0.05,
                            num_steps=3, device=None, progress=True):
    # No interneurons here: the function dimension equals the network size.
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid = [(n, p) for n in n_vals for p in p_vals]
    rows = []
    sink = io.StringIO()
    for n, p in tqdm(grid, disable=not progress, desc="small-network sweep"):
        for _ in range(nets_per_config):
            with contextlib.redirect_stdout(sink):
                W = sample_network(network_model, n, p)
            num_edges = int(W.sum())
            density = W.sum() / (n ** 2 - n)
            num_cycles = int(get_num_cycles(W))
            for _ in range(funcs_per_net):
                TT = get_random_TT_from_coin_flips(n, 0.5)
                with contextlib.redirect_stdout(sink):
                    accuracy = train_one(n, n, W, TT, n_seeds=n_seeds, epochs=epochs,
                                         lr=lr, num_steps=num_steps, device=device)
                rows.append({"n": n, "p": p, "density": density, "num_edges": num_edges,
                             "num_cycles": num_cycles, "accuracy": accuracy,
                             "solves": int(accuracy == 1)})
    return pd.DataFrame(rows)


def count_3_cycles_sweep(sizes, p_vals, n_networks=100, network_model="ER", progress=True):
    rows = []
    grid = [(n, p) for n in sizes for p in p_vals]
    for n, p in tqdm(grid, disable=not progress, desc="3-cycle counts"):
        for _ in range(n_networks):
            W = sample_network(network_model, n, p)
            rows.append({
                "n": n,
                "p": p,
                "network_model": network_model,
                "num_edges": int(W.sum()),
                "density": W.sum() / (n ** 2 - n),
                "num_cycles": int(get_num_cycles(W)),
            })
    return pd.DataFrame(rows)
