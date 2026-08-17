import sys
import time
import pickle
import os
import uuid
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

from utils.topology import construct_topology_from_int, topology_to_motifs_vector
from utils.truth_tables import get_truth_table_by_idx
from utils.nn_models import FF_2Layer, perform_NetSolvesFunc_prediction
from utils.catalog_matrix import get_catalog_matrix_with_indices

def generate_data_split_for_NetFunc_predictions(n, catalog_matrix, approximation_matrix, num_functions=None, num_networks=None):
    if num_functions is not None and num_networks is not None:
        print(f"Loaded catalog matrix with shape {catalog_matrix.shape} and approximation matrix with shape {approximation_matrix.shape}")
        print(f"Sampling {num_networks} networks and {num_functions} functions for n={n}")
        sampled_networks_indices = np.random.choice(list(catalog_matrix.index), size=num_networks, replace=False)
        sampled_functions_indices = np.random.choice(list(catalog_matrix.columns), size=num_functions, replace=False)
        catalog_matrix = catalog_matrix.loc[sampled_networks_indices, sampled_functions_indices]
        approximation_matrix = approximation_matrix.loc[sampled_networks_indices, sampled_functions_indices]
        print(f"Sampled {num_networks} networks and {num_functions} functions for n={n}")
    else:
        print(f"Using all networks and functions for with size {catalog_matrix.shape} for n={n}")

    network_indices = np.array(catalog_matrix.index)
    function_indices = np.array(catalog_matrix.columns)

    np.random.shuffle(network_indices)
    np.random.shuffle(function_indices)

    train_portion = 0.5
    networks_train_size = int(train_portion * len(network_indices))
    functions_train_size = int(train_portion * len(function_indices))

    print(f"With train portion of {train_portion}, we have {networks_train_size} networks and {functions_train_size} functions for training")

    networks_train_indices = network_indices[:networks_train_size]
    networks_test_indices = network_indices[networks_train_size:]

    functions_train_indices = function_indices[:functions_train_size]
    functions_test_indices = function_indices[functions_train_size:]

    X_train_data_connectivity_matrix = []
    X_train_data_motifs = []
    X_train_data_num_edges = []
    X_train_data_num_reciprocal_edges = []
    X_train_data_num_3_cycles = []
    X_train_data_2_3_cycles = []
    X_train_data_edges_2_3_cycles = []
    X_train_data_motifs_connectivity = []

    X_test_data_connectivity_matrix = []
    X_test_data_motifs = []
    X_test_data_num_edges = []
    X_test_data_num_reciprocal_edges = []
    X_test_data_num_3_cycles = []
    X_test_data_2_3_cycles = []
    X_test_data_edges_2_3_cycles = []
    X_test_data_motifs_connectivity = []

    y_binary_train = []
    y_binary_test = []
    y_accuracy_train = []
    y_accuracy_test = []

    print(f"\tWorking on TRAIN data")
    for i, network_idx in enumerate(networks_train_indices):
        topo = construct_topology_from_int(network_idx, n)

        network_as_connectivity_matrix = topo.flatten().astype(int)
        network_as_motifs_vector = topology_to_motifs_vector(topo, n)
        network_as_num_edges = np.array([topo.sum()])
        network_as_num_reciprocal_edges = np.array([(topo * topo.T).sum() / 2])
        network_as_num_cycles = np.array([int(np.trace(topo @ topo @ topo) // 6)])

        for function_idx in functions_train_indices:
            TT = get_truth_table_by_idx(function_idx, n)
            flattened_TT = TT.table.reshape(TT.num_rows).astype(int)

            X_train_data_connectivity_matrix.append(np.concatenate([network_as_connectivity_matrix, flattened_TT]))
            X_train_data_motifs.append(np.concatenate([network_as_motifs_vector, flattened_TT]))
            X_train_data_num_edges.append(np.concatenate([network_as_num_edges, flattened_TT]))
            X_train_data_num_reciprocal_edges.append(np.concatenate([network_as_num_reciprocal_edges, flattened_TT]))
            X_train_data_num_3_cycles.append(np.concatenate([network_as_num_cycles, flattened_TT]))
            X_train_data_2_3_cycles.append(np.concatenate([network_as_num_reciprocal_edges, network_as_num_cycles, flattened_TT]))
            X_train_data_edges_2_3_cycles.append(np.concatenate([network_as_num_edges, network_as_num_reciprocal_edges, network_as_num_cycles, flattened_TT]))
            X_train_data_motifs_connectivity.append(np.concatenate([network_as_motifs_vector, network_as_connectivity_matrix, flattened_TT]))

            y_binary_train.append(catalog_matrix.loc[network_idx, function_idx])
            y_accuracy_train.append(approximation_matrix.loc[network_idx, function_idx])

        # print(f"Finished working TRAIN on network {i+1}/{len(networks_train_indices)}")

    print(f"\tWorking on TEST data")
    for i, network_idx in enumerate(networks_test_indices):
        topo = construct_topology_from_int(network_idx, n)

        network_as_connectivity_matrix = topo.flatten().astype(int)
        network_as_motifs_vector = topology_to_motifs_vector(topo, n)
        network_as_num_edges = np.array([topo.sum()])
        network_as_num_reciprocal_edges = np.array([(topo * topo.T).sum() / 2])
        network_as_num_cycles = np.array([int(np.trace(topo @ topo @ topo) // 6)])

        for function_idx in functions_test_indices:
            TT = get_truth_table_by_idx(function_idx, n)
            flattened_TT = TT.table.reshape(TT.num_rows).astype(int)

            X_test_data_connectivity_matrix.append(np.concatenate([network_as_connectivity_matrix, flattened_TT]))
            X_test_data_motifs.append(np.concatenate([network_as_motifs_vector, flattened_TT]))
            X_test_data_num_edges.append(np.concatenate([network_as_num_edges, flattened_TT]))
            X_test_data_num_reciprocal_edges.append(np.concatenate([network_as_num_reciprocal_edges, flattened_TT]))
            X_test_data_num_3_cycles.append(np.concatenate([network_as_num_cycles, flattened_TT]))
            X_test_data_2_3_cycles.append(np.concatenate([network_as_num_reciprocal_edges, network_as_num_cycles, flattened_TT]))
            X_test_data_edges_2_3_cycles.append(np.concatenate([network_as_num_edges, network_as_num_reciprocal_edges, network_as_num_cycles, flattened_TT]))
            X_test_data_motifs_connectivity.append(np.concatenate([network_as_motifs_vector, network_as_connectivity_matrix, flattened_TT]))

            y_binary_test.append(catalog_matrix.loc[network_idx, function_idx])
            y_accuracy_test.append(approximation_matrix.loc[network_idx, function_idx])

        # print(f"Finished working TEST on network {i+1}/{len(networks_test_indices)}")

    print(f"Train size {len(X_train_data_connectivity_matrix)}, Test size {len(X_test_data_connectivity_matrix)}")

    return {
        "X_train_data_connectivity_matrix": X_train_data_connectivity_matrix,
        "X_train_data_motifs": X_train_data_motifs,
        "X_train_data_num_edges": X_train_data_num_edges,
        "X_train_data_num_reciprocal_edges": X_train_data_num_reciprocal_edges,
        "X_train_data_num_3_cycles": X_train_data_num_3_cycles,
        "X_train_data_2_3_cycles": X_train_data_2_3_cycles,
        "X_train_data_edges_2_3_cycles": X_train_data_edges_2_3_cycles,
        "X_train_data_motifs_connectivity": X_train_data_motifs_connectivity,

        "X_test_data_connectivity_matrix": X_test_data_connectivity_matrix,
        "X_test_data_motifs": X_test_data_motifs,
        "X_test_data_num_edges": X_test_data_num_edges,
        "X_test_data_num_reciprocal_edges": X_test_data_num_reciprocal_edges,
        "X_test_data_num_3_cycles": X_test_data_num_3_cycles,
        "X_test_data_2_3_cycles": X_test_data_2_3_cycles,
        "X_test_data_edges_2_3_cycles": X_test_data_edges_2_3_cycles,
        "X_test_data_motifs_connectivity": X_test_data_motifs_connectivity,

        "y_binary_train": y_binary_train,
        "y_binary_test": y_binary_test,
        "y_accuracy_train": y_accuracy_train,
        "y_accuracy_test": y_accuracy_test,
    }

def train_NetSolvesFunc_classifiers(dim, num_nets_to_sample, num_funcs_to_sample, catalog_matrix_path, output_base_path, lr=0.01, epochs=100):
    start_time = time.time()

    approximation_matrix = get_catalog_matrix_with_indices(catalog_matrix_path)

    catalog_matrix = approximation_matrix.copy()
    catalog_matrix[catalog_matrix < 1] = 0

    if torch.cuda.is_available():
        dev = "cuda:0"
        print(f"Found a CUDA device - {torch.cuda.get_device_name(0)}")
    else:
        print("\tNo GPU found. Working on CPU\n")
        dev = "cpu"

    sys.stdout.flush()
    sys.stderr.flush()

    device = torch.device(dev)

    experiments_per_model = defaultdict(list)

    t1 = time.time()
    data_split_NetFunc = generate_data_split_for_NetFunc_predictions(n=dim,
                                                                    catalog_matrix=catalog_matrix,
                                                                    approximation_matrix=approximation_matrix,
                                                                    num_functions=num_funcs_to_sample,
                                                                    num_networks=num_nets_to_sample)

    t2 = time.time()
    print(f"Data split done in {t2-t1} seconds")
    sys.stdout.flush()
    sys.stderr.flush()

    y_accuracy_train = data_split_NetFunc["y_accuracy_train"]
    y_accuracy_test = data_split_NetFunc["y_accuracy_test"]

    y_binary_train = data_split_NetFunc["y_binary_train"]
    y_binary_test = data_split_NetFunc["y_binary_test"]

    experiments = [
        ("ConnectivityMatrix",
            {"X_train": data_split_NetFunc["X_train_data_connectivity_matrix"],
            "X_test": data_split_NetFunc["X_test_data_connectivity_matrix"]
        }),
        ("Motifs",
            {"X_train": data_split_NetFunc["X_train_data_motifs"],
            "X_test": data_split_NetFunc["X_test_data_motifs"]
        }),
        ("#edges",
            {"X_train": data_split_NetFunc["X_train_data_num_edges"],
            "X_test": data_split_NetFunc["X_test_data_num_edges"]
        }),
        ("#2-cycles",
            {"X_train": data_split_NetFunc["X_train_data_num_reciprocal_edges"],
            "X_test": data_split_NetFunc["X_test_data_num_reciprocal_edges"]
        }),
        ("#3-cycles",
            {"X_train": data_split_NetFunc["X_train_data_num_3_cycles"],
            "X_test": data_split_NetFunc["X_test_data_num_3_cycles"]
        }),
        ("#edges+cycles",
            {"X_train": data_split_NetFunc["X_train_data_edges_2_3_cycles"],
            "X_test": data_split_NetFunc["X_test_data_edges_2_3_cycles"]
        }),
        ("Connectivity+Motifs",
         {"X_train": data_split_NetFunc["X_train_data_motifs_connectivity"],
          "X_test": data_split_NetFunc["X_test_data_motifs_connectivity"]}
         )
    ]

    # models_to_test = [("linear_reg", LinReg), ("FF_1Layer", FF_1Layer), ("FF_2Layer", FF_2Layer), ("DeepFF", DeepFF)]
    # models_to_test = [("FF_1Layer", FF_1Layer)]
    models_to_test = [("FF_2Layer", FF_2Layer)]

    for model_name, model_class in models_to_test:
        print(f"\n \t################## WORKING ON MODEL {model_name} ################## \n")


        for experiment_name, experiment_metadata in experiments:
            print(f"\tWorking on {experiment_name}")
            X_train = experiment_metadata["X_train"]
            X_test = experiment_metadata["X_test"]

            print(f"\t Running an experiment with {model_name} on {experiment_name}")
            t1 = time.time()
            opt_results_binary = perform_NetSolvesFunc_prediction(
                model_class=model_class,
                criterion=nn.BCEWithLogitsLoss(),
                representation_dim=X_train[0].shape[0],
                X_train=X_train,
                X_test=X_test,
                y_train=y_binary_train,
                y_test=y_binary_test,
                lr=lr,
                num_epochs=epochs,
                batch_size=64,
                device=device
            )

            experiments_per_model[model_name].append({
                "experiment_name": experiment_name,
                "y": "binary",
                "accuracy": opt_results_binary["accuracy"],
                "precision": opt_results_binary["precision"],
                "recall": opt_results_binary["recall"],
                "f1": opt_results_binary["f1"],
                "roc_auc": opt_results_binary["roc_auc"],

                })
            t2 = time.time()
            print("\n")
            print(f"\t\tDone in {t2-t1} seconds")
            sys.stdout.flush()
            sys.stderr.flush()

    # results_path = f"n_{dim}__n_nets_{num_nets_to_sample}__n_funcs_{num_funcs_to_sample}"
    results_path = f"NetSolvesFunc__n_{dim}"
    results_path = os.path.join(output_base_path, results_path)
    Path(results_path).mkdir(parents=True, exist_ok=True)
    print(f"Writing results to `{results_path}`")

    random_job_id = str(uuid.uuid4())[-6:]
    output_path = os.path.join(results_path, f"n_nets_{num_nets_to_sample}__n_funcs_{num_funcs_to_sample}__{random_job_id}.pkl")
    print(f"Writing results to {output_path}")
    with open(output_path, "wb") as f:
        output_data = {
            "experiments_per_model": experiments_per_model,
            "dim": dim,
            "num_nets_to_sample": num_nets_to_sample,
            "num_funcs_to_sample": num_funcs_to_sample,
            "lr": lr,
            "epochs": epochs
        }
        pickle.dump(output_data, f)

    end_time = time.time()
    print(f"DONE. In total, everything took {end_time-start_time} seconds")
