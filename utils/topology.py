import math
import numpy as np
import networkx as nx
import itertools
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def construct_topology_from_int(int_code, size, auto_synapses=False):

    max_num_of_nets = 2 ** (size ** 2 - size)
    if int_code >= max_num_of_nets:
        raise ValueError(f"{int_code} is too large for network size {size} (max={max_num_of_nets})")

    topology_str = f'{int_code:0{(size ** 2 - size)}b}'
    topology_no_diag = np.array(list(topology_str), 'uint8')
    topology = np.zeros(size ** 2)
    for k in range(size - 1):
        topology[k * (size + 1) + 1: (k + 1) * (size + 1)] = topology_no_diag[k * size: (k + 1) * size]
    topology = topology.reshape((size, size))

    if auto_synapses:
        np.fill_diagonal(topology, 1)

    return topology

def convert_topology_to_int(topology):
    diag_indices = np.diag_indices_from(topology)
    flattened_matrix = np.delete(topology, np.ravel_multi_index(diag_indices, topology.shape))
    flattened_matrix = np.array(flattened_matrix, dtype=int)
    return int(''.join(map(str, flattened_matrix)), 2)


def get_all_topologies(network_size):
    """
    Create a generator that yields all possible topologies for a given network size
    """
    num_topologies = 2 ** (network_size ** 2 - network_size)

    for i in range(num_topologies):
        topology = construct_topology_from_int(i, network_size)
        yield topology

def generate_random_topology(size, real_weights=False):
    cur_topology = np.random.randint(0, 2, size=(size, size))
    np.fill_diagonal(cur_topology, 0)
    if real_weights:
        cur_topology = cur_topology * np.random.randn(size, size)
    return cur_topology

def permute_topology(W, perm):
    return W[np.ix_(perm, perm)]

def get_num_cycles(W: np.ndarray):
    W3 = W @ W @ W
    num_3_cycles = np.trace(W3) // 3
    return num_3_cycles

def get_num_connected_components(W: np.ndarray):
    """
    Count the number of weakly connected components in a directed graph represented by its adjacency matrix W.
    We convert the directed graph to an undirected graph by adding the transpose of W, and then use networkx to count the connected components.
    """
    undirected_G = (W + W.T) > 0
    G = nx.from_numpy_array(undirected_G)
    return len(list(nx.connected_components(G)))

def get_3_reachability(W):
    """
    Calculate the number of nodes reachable in 3 steps from each node in the directed graph represented by the adjacency matrix W.
    """
    W2 = W @ W
    W2[np.eye(W2.shape[0], dtype=bool)] = 0
    W3 = W2 @ W
    W3[np.eye(W3.shape[0], dtype=bool)] = 0

    reachability_matrix = W + W2 + W3
    reachability_matrix[reachability_matrix > 0] = 1

    return np.sum(reachability_matrix, axis=0)

def motif_name_to_idx(motif_name):
    """
    Convert a motif name to an index in the motifs vector. We have 16 motifs, so the index is in [0, 15].
    Motif names are according to https://networkx.org/documentation/stable/auto_examples/graph/plot_triad_types.html
    """
    triads_to_idx = {
        "003": 0,
        "012": 1,
        "102": 2,
        "021D": 3,
        "021U": 4,
        "021C": 5,
        "111D": 6,
        "111U": 7,
        "030T": 8,
        "030C": 9,
        "201": 10,
        "120D": 11,
        "120U": 12,
        "120C": 13,
        "210": 14,
        "300": 15
    }

    return triads_to_idx[motif_name]

def idx_to_motif_name(idx):
    idx_to_triad = {
        0: "003",
        1: "012",
        2: "102",
        3: "021D",
        4: "021U",
        5: "021C",
        6: "111D",
        7: "111U",
        8: "030T",
        9: "030C",
        10: "201",
        11: "120D",
        12: "120U",
        13: "120C",
        14: "210",
        15: "300"
    }

    return idx_to_triad[idx]

def take_subgraph_from_topology(topology, nodes):
    """
    Receive a topology and return the subgraph that contains only the nodes in the list `nodes`.
    """
    only_rows = topology[nodes, :]
    subgraph = only_rows[:, nodes]

    return subgraph


def topology_to_motifs_vector(topology, dim):
    """
    Receive a n x n topology matrix and return a 16-dim vector of motifs
    """
    motifs = np.zeros(16)

    for comb in itertools.combinations(range(dim), 3):
        motif_topo = take_subgraph_from_topology(topology, comb)
        motif_name = topology_to_triad(motif_topo, False)
        motifs[motif_name_to_idx(motif_name)] += 1

    return motifs

def topology_to_triad(topology, is_networkx_topology):
    """
    For us, entry i,j in topology matrix indicates that node i receives an input from node j!
    So W[i,j] = 1  means that j -> i

    * networkx works the opposite, so we need to transpose the matrix.
    * Our own `construct_topology_from_int` function creates correct matrices, but they need to be transposed for networkx.
    """
    import networkx as nx

    topo = topology.T if not is_networkx_topology else topology

    G = nx.DiGraph(topo)
    return nx.triad_type(G)

def nets_to_equivalence_classes(dim, output_path=None):
    """
    Iterate over all 2^n(n-1) possible network architectures, and divide them into
    equivalence classes, in which all networks are equivalent up to node permutation.

    Return
    -------
        equiv_classes: list
            A list of lists, where each inner list contains all networks in an equivalence class.
    """
    all_permutations = list(itertools.permutations(range(dim)))
    equivalence_classes_indices = []
    equivalence_classes = []

    num_of_nets = 2 ** (dim ** 2 - dim)

    for net_idx in range(1, num_of_nets):
        print(f"Working on net_idx # {net_idx}")

        topo = construct_topology_from_int(net_idx, dim)

        if len(equivalence_classes) == 0:
            equivalence_classes.append([net_idx])
            equivalence_classes_indices.append(net_idx)
            continue

        found_match = False

        num_of_edges = np.sum(topo)

        for j, equiv_class_representative_idx in enumerate(equivalence_classes_indices):

            representative_num_edges = np.sum(construct_topology_from_int(equiv_class_representative_idx, dim))
            if num_of_edges != representative_num_edges:
                continue

            for perm in all_permutations:
                permed_topo = permute_topology(topo, perm)
                permed_topo_idx = convert_topology_to_int(permed_topo)
                if permed_topo_idx == equiv_class_representative_idx:
                    print(f"\t\tFound match for net_idx {net_idx} in equiv class {j}")

                    equivalence_classes[j].append(net_idx)
                    found_match = True
                    break

            if found_match:
                break

        # Open a new equiv. class
        if not found_match:
            equivalence_classes.append([net_idx])
            equivalence_classes_indices.append(net_idx)
            print(f"\t\tCreating a new class for net # {net_idx}. Total classes = {len(equivalence_classes)}")


    print(f"Found {len(equivalence_classes)} classes for n={dim}!")
    print("#### ")

    if output_path is not None:
        print(f"Saving the equivalence classes to {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(equivalence_classes, f)
        print("#### DONE")

    return equivalence_classes

def get_random_overlapping_cycles_nets(n, p):
    num_edges = np.ceil(np.random.normal(n*(n-1)*p, 2))
    empty_net = np.zeros((n, n))

    i, j, k = np.random.choice(n, 3, replace=False)

    empty_net[i, j] = 1
    empty_net[j, k] = 1
    empty_net[k, i] = 1

    W = empty_net.copy()

    num_edges -= 3

    while num_edges > 0:
        # First try creating a completeley new cycle -
        i, j, k = np.random.choice(n, 3, replace=False)

        if W[i, j] == 0 and W[j, k] == 0 and W[k, i] == 0:
            W[i, j] = 1
            W[j, k] = 1
            W[k, i] = 1
            num_edges -= 3
            continue

        len_2_paths = W @ W # The i, j entry in W @ W gives the number of 2-paths from i to j
        np.fill_diagonal(len_2_paths, 0) # closing a 2-path i->k->i would add a self connection
        existing_2_path_i_j = np.nonzero(len_2_paths)

        single_edge_cycle_candidates = []

        for edge in zip(*existing_2_path_i_j):
            i, j = edge
            if W[j, i] == 0:
                single_edge_cycle_candidates.append((j, i))

        if len(single_edge_cycle_candidates) > 0:
            random_edge_idx = np.random.choice(range(len(single_edge_cycle_candidates)))
            edge_to_add = single_edge_cycle_candidates[random_edge_idx]
            W[edge_to_add] = 1
            num_edges -= 1

            # print(f"Added a cycle by adding a single edge: {edge_to_add}")
        elif num_edges > 2: # Must add a cycle by adding two edges. Also verify there are enough edges left
            existing_edges = [(i, j) for i, j in zip(*existing_2_path_i_j)]
            random_edge_idx = np.random.choice(range(len(existing_edges)))
            dest, src = existing_edges[random_edge_idx]

            # print(f"\t selected: {(dest+1, src+1)}")

            # Pick a random node to connect to -
            random_node = np.random.choice(n)
            while random_node in (dest, src) or W[random_node, src] == 1 or W[dest, random_node] == 1:
                random_node = np.random.choice(n)

            # print(f"\t random node: {random_node+1}")

            W[random_node, src] = 1
            W[dest, random_node] = 1

            num_edges -= 2

            # print(f"Added a cycle by adding two edges: {src+1} -> {random_node+1} -> {dest+1}")

        else: # Just turn a random edge on
            i, j = np.random.choice(n, 2, replace=False)
            while W[i, j] == 1:
                i, j = np.random.choice(n, 2, replace=False)

            W[i, j] = 1
            num_edges -= 1
            # print(f"Added a random edge: {i+1} -> {j+1}")

    # print(W)
    return W

def get_p_sparse_DAG(n, p):
    """
    Return a random directed acyclic graph (DAG) with n nodes and a density of p.
    The graph is constructed by first building a complete directed acyclic graph (MaxDAG) and then pruning edges to achieve the desired density.

    MaxDAG is basically a "Transitive Tournament" graph.
    """
    W = np.triu(np.ones((n, n)), k=1)

    max_num_edges = n * (n-1)
    desired_num_edges = round(max_num_edges * p)
    current_num_edges = max_num_edges // 2 # W is MaxDAG, i.e. a graph with density of 50%

    num_edges_to_prune = current_num_edges - desired_num_edges

    possible_edges = np.argwhere(W == 1)
    edge_idx_to_prune = np.random.choice(possible_edges.shape[0], size=num_edges_to_prune, replace=False)
    edges_to_prune = possible_edges[edge_idx_to_prune]
    W[edges_to_prune[:, 0], edges_to_prune[:, 1]] = 0

    return W

def get_p_sparse_DAG_random_ordering(n, p):
    W = get_p_sparse_DAG(n, p)
    rng = np.random.default_rng(None)
    perm = rng.permutation(n)
    W = W[perm][:, perm]
    return W

def farthest_pair(W):
    """
    Given an n x n directed connectivity matrix W (0/1),
    return a pair (u, v) such that:
      - either u cannot reach v (unreachable), or
      - the distance from u to v is maximal among all reachable pairs
    """
    n = W.shape[0]
    graph = csr_matrix(W)  # Convert to sparse for performance
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=True, return_predecessors=True)

    # Check for unreachable pairs (infinite distance)
    if np.isinf(dist_matrix).any():
        i, j = np.where(np.isinf(dist_matrix))
        return (i[0], j[0])  # Return the first unreachable pair
    else:
        # All pairs are reachable; return the pair with maximum distance
        i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        return (i, j)

def find_3_cycle(W):
    n = W.shape[0]
    for i in range(n):
        js = np.where(W[i])[0]
        for j in js:
            ks = np.where(W[j])[0]
            for k in ks:
                if W[k, i]:
                    return (i, j, k)
    return None

def get_random_i_j(W):
    n = W.shape[0]
    while True:
        i, j = np.random.randint(0, n, size=2)
        if W[i, j] == 0:  # Ensure the edge does not already exist
            return i, j

def get_reachable_no_cycles_mask(n, p):
    max_n_iters = math.comb(n, 3)
    iterations = 0

    W = nx.to_numpy_array(nx.erdos_renyi_graph(n, p, directed=True))

    num_3_cycles = get_num_cycles(W)
    print(f"[{iterations}] num_3_cycles: {num_3_cycles}")

    reachability = get_3_reachability(W)
    # print(f"[{iterations}] reachability: {reachability}")
    normalized_reachability = [x / n for x in reachability]
    mean_reachability = np.mean(normalized_reachability)
    print(f"\tmean normalized reachability: {mean_reachability:.3f}")

    iters = 0
    while True:
        if iters >= max_n_iters:
            print(f"Reached maximum iterations ({max_n_iters}), stopping.")
            break
        iters += 1
        cycle_indices = find_3_cycle(W)
        if cycle_indices is None:
            print("No more 3-cycles found, stopping.")
            break
        i, j = cycle_indices[0], cycle_indices[1]

        W[i, j] = 0

        new_i, new_j = farthest_pair(W)
        if new_i == i and new_j == j:
            # print("Farthest pair is the same as the removed edge, stopping.")
            new_i, new_j = get_random_i_j(W)

        W[new_i, new_j] = 1

        # print(f"[{iterations+1}] Removed edges ({i},{j})")
        # print(f"[{iterations+1}] Added edges ({new_i},{new_j})")

        num_3_cycles = get_num_cycles(W)
        # print(f"\tnum_3_cycles: {num_3_cycles}")
        reachability = get_3_reachability(W)
        normalized_reachability = [x / n for x in reachability]
        mean_reachability = np.mean(normalized_reachability)
        # print(f"\tmean normalized reachability: {mean_reachability:.3f}")
        num_edges = np.sum(W)
        # print(f"\t|E| = {num_edges}")

        iterations += 1

    return W
