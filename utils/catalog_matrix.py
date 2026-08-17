import pickle
import re
import uuid
from collections import defaultdict

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

from utils.topology import construct_topology_from_int
from utils.truth_tables import get_truth_table_by_idx, TT_to_fourier_expansion, get_max_degree_in_fourier_expansion, calculate_total_influence_for_TT, calculate_num_of_zero_inf_bits

def sort_cat_mat_by_edges(cat_mat, dim, return_type="np"):
    catmat_as_df = pd.DataFrame(cat_mat).copy()
    catmat_as_df["num_of_edges"] = catmat_as_df.apply(lambda x: np.sum(construct_topology_from_int(x.name, dim, auto_synapses=False)), axis=1)
    sorted_catmat_by_edges_df = catmat_as_df.sort_values(by=["num_of_edges"])
    sorted_catmat_by_edges_df.drop("num_of_edges", axis=1, inplace=True)

    if return_type == "np":
        return np.array(sorted_catmat_by_edges_df)
    elif return_type == "df":
        return sorted_catmat_by_edges_df

    return sorted_catmat_by_edges_df

def sort_cat_mat_by_alpha(cat_mat, dim, return_type="np"):
    catmat_as_df = pd.DataFrame(cat_mat).copy()

    binary_cat_mat = catmat_as_df.copy()
    binary_cat_mat[binary_cat_mat < 1] = 0

    # alpha = catmat_as_df.sum(axis=1) / catmat_as_df.shape[1]
    alpha = binary_cat_mat.sum(axis=1) / binary_cat_mat.shape[1]

    catmat_as_df["alpha"] = alpha
    sorted_catmat_by_edges_df = catmat_as_df.sort_values(by=["alpha"])
    sorted_catmat_by_edges_df.drop("alpha", axis=1, inplace=True)

    if return_type == "np":
        return np.array(sorted_catmat_by_edges_df)
    elif return_type == "df":
        return sorted_catmat_by_edges_df

    return sorted_catmat_by_edges_df

def sort_cat_mat_by_max_boolean_func_criteria(cat_mat, dim, criteria):
    def _get_max_FE_deg(func_idx):
        TT = get_truth_table_by_idx(func_idx, dim)
        fourier_expansion = TT_to_fourier_expansion(TT)
        return get_max_degree_in_fourier_expansion(fourier_expansion)

    def _get_total_influence(func_idx):
        TT = get_truth_table_by_idx(func_idx, dim)
        return calculate_total_influence_for_TT(TT)

    def _get_zero_inf_bits_count(func_idx):
        TT = get_truth_table_by_idx(func_idx, dim)
        num_of_ZIB = calculate_num_of_zero_inf_bits(TT)
        return dim - num_of_ZIB

    cat_mat = cat_mat.copy().T.rename_axis("func_idx", axis=1)
    if criteria == "max_FE_deg":
        cat_mat["sort_criteria"] = cat_mat.apply(lambda x: _get_max_FE_deg(x.name), axis=1)
    elif criteria == "total_influence":
        cat_mat["sort_criteria"] = cat_mat.apply(lambda x: _get_total_influence(x.name), axis=1)
    elif criteria == "ZIB":
        cat_mat["sort_criteria"] = cat_mat.apply(lambda x: _get_zero_inf_bits_count(x.name), axis=1)
    else:
        raise ValueError(f"Unknown criteria {criteria}")

    sorted_cat_mat = cat_mat.sort_values(by=["sort_criteria"], ascending=False)
    sorted_cat_mat.drop("sort_criteria", axis=1, inplace=True)
    sorted_cat_mat = sorted_cat_mat.T.rename_axis("network_idx", axis=1)

    return sorted_cat_mat

def sort_columns_by_num_of_solving_networks(cat_mat, binarize=False):
    cat_mat = cat_mat.copy().T.rename_axis("func_idx", axis=1)

    if binarize:
        cat_mat_for_indices = cat_mat.copy()
        cat_mat_for_indices[cat_mat_for_indices < 1] = 0
    else:
        cat_mat_for_indices = cat_mat.copy()

    num_of_solvers_per_function = cat_mat_for_indices.sum(axis=1)
    sorted_indices = np.argsort(num_of_solvers_per_function)

    return cat_mat.iloc[sorted_indices].T.rename_axis("network_idx", axis=1)

def run_agglomerative_clustering_for_df_by_columns_and_by_total_solvers(cat_mat, n_clusters, linkage="average", distance_metric="hamming", magic3=False, magic4=False):
    cat_mat = cat_mat.copy().T.rename_axis("func_idx", axis=1)
    distance_matrix = pdist(cat_mat, distance_metric)
    distance_matrix = squareform(distance_matrix)

    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage).fit(distance_matrix)

    cat_mat_with_label = cat_mat.copy()
    cat_mat_with_label["cluster_label"] = clustering.labels_
    cat_mat_with_label["total_solvers_per_function"] = cat_mat_with_label.sum(axis=1) / cat_mat_with_label.shape[1]
    avg_solved_per_cluster = cat_mat_with_label.groupby("cluster_label").mean()["total_solvers_per_function"].round()
    cat_mat_with_label["cluster_label_as_mean"] = avg_solved_per_cluster[clustering.labels_].values
    sorted_cat_mat = cat_mat_with_label.sort_values(["cluster_label_as_mean", "total_solvers_per_function"], ascending=[True, True])
    sorted_cat_mat = sorted_cat_mat.drop(["cluster_label", "total_solvers_per_function", "cluster_label_as_mean"], axis=1)

    return sorted_cat_mat.T.rename_axis("network_idx", axis=1)

def get_catalog_matrix(path):
    with open(path, "rb") as f:
        catalog_matrix = pickle.load(f)

    return catalog_matrix

def get_catalog_matrix_with_indices(path):
    """
    Returns a DataFrame with the indices of the catalog matrix.
    Rows have the idx of the networks, columns are the indices of the functions
    """
    catalog_matrix = get_catalog_matrix(path)
    if not isinstance(catalog_matrix, pd.DataFrame):
        catalog_matrix = pd.DataFrame(catalog_matrix)
        catalog_matrix.index += 1

    catalog_matrix.index.name = "network_idx"

    return catalog_matrix

def compute_utility(catalog_matrix):
    return catalog_matrix.mean(axis=1)

def compute_mean_accuracy(approximation_matrix):
    return approximation_matrix.mean(axis=1)

def hamming_dist(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    return np.count_nonzero(v1 != v2) / v1.shape[0]

def create_catalog_matrix_plot(
                               is_binary,
                               nets_sorting_method,
                               funcs_sorting_method,
                               input_path=None,
                               catalog_matrix=None,
                               save_plot=True,
                               return_data=False,
                               **kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt

    random_state = kwargs.get("random_state", 0)

    MAX_NETS_SIZE = kwargs.get("max_nets_size", 256)
    MAX_FUNCS_SIZE = kwargs.get("max_funcs_size", 256)



    if catalog_matrix is not None:
        catalog_matrix = catalog_matrix.copy()
        n = kwargs.get("n")
        runtime = kwargs.get("runtime")
    elif input_path is not None:
        print(f"Working on {input_path}")
        pattern = r'^.+n_(\d+)__runtime_(\d+)_(\w+)'
        match = re.match(pattern, input_path)
        n = int(match.group(1))
        runtime = int(match.group(2))
        catalog_matrix = get_catalog_matrix_with_indices(input_path)
    else:
        raise ValueError("Either `input_path` or `catalog_matrix` must be provided")
    print(f"\t n = {n}, runtime = {runtime}")
    print(f"\tCatalog matrix shape - {catalog_matrix.shape}")

    if is_binary:
        catalog_matrix[catalog_matrix < 1] = 0

    # Randomly pick `MAX_SIZE` rows and columns -
    max_nets = min(MAX_NETS_SIZE, catalog_matrix.shape[0])
    max_funcs = min(MAX_FUNCS_SIZE, catalog_matrix.shape[1])
    catalog_matrix = catalog_matrix.sample(n=max_nets, axis=0, random_state=random_state).sample(n=max_funcs, axis=1, random_state=random_state)

    if nets_sorting_method == "edges":
        row_sorted_cat_mat = sort_cat_mat_by_edges(catalog_matrix, dim=n, return_type="df")
        row_sorting_str = "# of edges"
    elif nets_sorting_method == "alpha":
        row_sorted_cat_mat = sort_cat_mat_by_alpha(catalog_matrix, dim=n, return_type="df")
        row_sorting_str = "alpha (solving rate)"
    else:
        raise ValueError(f"Sorting method for networks {nets_sorting_method} not implemented")

    if funcs_sorting_method == "total_solving_nets_count":
        # column_sorted_cat_mat = sort_cat_mat_columns_by_total_solving_nets(row_sorted_cat_mat)
        column_sorted_cat_mat = sort_columns_by_num_of_solving_networks(row_sorted_cat_mat, binarize=True)

        col_sorting_str = "# of solving networks"
    elif funcs_sorting_method == "fourier_degree":
        column_sorted_cat_mat = sort_cat_mat_by_max_boolean_func_criteria(row_sorted_cat_mat, n, "max_FE_deg")
        col_sorting_str = "Max Fourier degree"
    elif funcs_sorting_method == "total_influence":
        column_sorted_cat_mat = sort_cat_mat_by_max_boolean_func_criteria(row_sorted_cat_mat, n, "total_influence")
        col_sorting_str = "Total influence"
    elif funcs_sorting_method == "num_zero_influence_bits":
        column_sorted_cat_mat = sort_cat_mat_by_max_boolean_func_criteria(row_sorted_cat_mat, n, "ZIB")
        col_sorting_str = "Num of zero influence bits"
    elif funcs_sorting_method == "agglomerative_clustering":
        n_clusters = kwargs.get("n_clusters", 4)
        distance_metric = "hamming"
        column_sorted_cat_mat = run_agglomerative_clustering_for_df(row_sorted_cat_mat, n_clusters=n_clusters, distance_metric=distance_metric)
        col_sorting_str = f"Agglomerative clustering ({distance_metric}, {n_clusters} clusters)"
    elif funcs_sorting_method == "clustering_and_solvers":

        n_clusters = kwargs.get("n_clusters", 4)
        distance_metric = "hamming"
        column_sorted_cat_mat = run_agglomerative_clustering_for_df_by_columns_and_by_total_solvers(row_sorted_cat_mat, n_clusters=n_clusters, distance_metric=distance_metric)
        col_sorting_str = f"Agglomerative clustering ({distance_metric}, {n_clusters} clusters)"

    else:
        raise ValueError(f"Sorting method for functions {funcs_sorting_method} not implemented")

    sorted_matrix = column_sorted_cat_mat.copy()

    if return_data:
        return sorted_matrix

    fig, ax = plt.subplots(figsize=(20, 12))
    fig = sns.heatmap(sorted_matrix, annot=False, linewidth=.5, fmt="g", cmap="Purples", annot_kws={"size": 8}, xticklabels=False, yticklabels=False)

    ax.set_title(f"Catalog matrix for n={n}, runtime={runtime} of size ({max_nets},{max_funcs}). \n Rows sorted by {row_sorting_str} \n Columns sorted by {col_sorting_str} \n {input_path}")
    ax.set(xlabel='Functions', ylabel='Networks')
    ax.tick_params(labelsize=8)
    tick_positions = range(0, len(sorted_matrix.columns), 15)
    tick_labels = sorted_matrix.columns[::15]
    plt.xticks(tick_positions, tick_labels, rotation=-45)

    tick_positions = range(1, len(sorted_matrix.index), 15)
    tick_labels = sorted_matrix.index[::15]
    plt.yticks(tick_positions, tick_labels)


    if save_plot:
        id = str(uuid.uuid4())[-6:]

        if is_binary:
            out_path = f"plots/n_{n}_runtime_{runtime}_binary_{id}.png"
        else:
            out_path = f"plots/n_{n}_runtime_{runtime}_nonbinary_{id}.png"
        print(f"Saving plot to {out_path}")
        plt.savefig(out_path)
    else:
        plt.show()

def reduce_to_equivalence_classes(catalog_matrix, net_equiv_classes, func_equiv_classes):
    reduced = np.zeros((len(net_equiv_classes), len(func_equiv_classes)))
    for i, net_class in enumerate(net_equiv_classes):
        for j, func_class in enumerate(func_equiv_classes):
            if catalog_matrix[np.ix_(net_class, func_class)].sum() > 0:
                reduced[i, j] = 1
    return reduced

def build_hierarchy_graph(net_equiv_classes, dim):
    graph = nx.DiGraph()
    representative_topologies = []
    for i, equiv_class in enumerate(net_equiv_classes):
        representative = equiv_class[0]
        topology = construct_topology_from_int(representative, dim)
        graph.add_node(i, members=equiv_class, num_edges=int(np.sum(topology)), example_topology=topology)
        representative_topologies.append(topology)

    for i, equiv_class_i in enumerate(net_equiv_classes):
        num_edges_i = representative_topologies[i].sum()
        for j, equiv_class_j in enumerate(net_equiv_classes):
            if i == j:
                continue
            for repr_j_idx in equiv_class_j:
                topology_j = construct_topology_from_int(repr_j_idx, dim)
                if num_edges_i >= topology_j.sum():
                    continue
                if np.sum(np.abs(topology_j - representative_topologies[i])) == 1:
                    graph.add_edge(i, j)
                    break
    return graph

def compute_minimal_solvers(reduced_catalog_matrix, hierarchy_graph):
    reverse_graph = hierarchy_graph.reverse()
    minimal_solvers = defaultdict(set)
    for func_class_idx in range(reduced_catalog_matrix.shape[1]):
        for node_idx in reverse_graph.nodes():
            if reduced_catalog_matrix[node_idx, func_class_idx] < 1:
                continue
            children = list(reverse_graph.successors(node_idx))
            children_accuracies = [reduced_catalog_matrix[c, func_class_idx] for c in children]
            if all(acc < 1 for acc in children_accuracies):
                minimal_solvers[node_idx].add(func_class_idx)
    return minimal_solvers
