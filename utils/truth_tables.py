from core.truth_table import TruthTable
import numpy as np
import itertools
import pickle
import sys

def get_all_truth_tables(dim):
    """
    Generate all truth tables for f:{0,1}^dim --> {0,1}.

    We generate all binary strings of length 2^n. Each one represents the outputs of a truth table.
    """
    for i in range(2**2**dim):
        print(f"Working on i={i}")
        tt_outputs = f"{i:0{2**dim}b}"
        print(f"\tOutputs = {tt_outputs}")
        truth_table = TruthTable(dim, 1, is_random=False)

        for output_idx, output_bit in enumerate(tt_outputs):
            truth_table.set_f_x(output_idx, int(output_bit))

        yield truth_table

def get_idx_from_truth_table(TT: TruthTable):
    """
    Convert a TT into an index (represented by an integer)
    """
    tt_outputs = [1 if x == 1 else 0 for x in TT.table]
    tt_outputs = "".join(map(str, tt_outputs))
    return int(tt_outputs, 2)

def get_truth_table_by_idx(idx: int, dim: int):
    """
    Convert a TT index (represented by an integer) into a TT
    """
    tt_outputs = f"{idx:0{2**dim}b}"
    truth_table = TruthTable(dim, 1, is_random=False)

    for output_idx, output_bit in enumerate(tt_outputs):
        truth_table.set_f_x(output_idx, int(output_bit))

    return truth_table

def get_random_TT_iterator(n, k):
    """
    Return an iterator of k boolean functions (truth tables) with dimension n.
    """

    # Using np.random.choice() with replace=False is very slow even for n=5... so using this simple implementation instead
    unique_set_of_indices = set()
    while len(unique_set_of_indices) < k:
        unique_set_of_indices.add(np.random.randint(0, 2**2**n))

    for i in unique_set_of_indices:
        yield (i, get_truth_table_by_idx(i, n))

def get_random_TT_from_coin_flips(n, coin_p):
    tt_outputs = np.random.binomial(1, p=coin_p, size=2**n)
    truth_table = TruthTable(n, 1, is_random=False)

    for output_idx, output_bit in enumerate(tt_outputs):
        truth_table.set_f_x(int(output_idx), int(output_bit))

    return truth_table

def get_random_TT_iterator_high_dim(n, k, sampling_method="uniform", binomial_p=0.25):
    """
    For n>=6 we can no longer even run np.random.randint(0, 2**2**n) since it
    exceeds int64. So instead we will simply create TT's by generating
    their 2^n outputs by flipping sequential coins
    """

    for i in range(k):
        if sampling_method == "uniform":
            tt_outputs = np.random.randint(0, 2, 2**n)
        elif sampling_method == "binomial":
            tt_outputs = np.random.binomial(1, p=binomial_p, size=2**n)
        truth_table = TruthTable(n, 1, is_random=False)

        for output_idx, output_bit in enumerate(tt_outputs):
            truth_table.set_f_x(int(output_idx), int(output_bit))

        yield i, truth_table

def get_random_k_sparse_parity_TT(n, k):
    """
    Create a random n-k parity function - a function on n bits where the output is the XOR of k randomly chosen bits.
    k = 0 is the constant 0 function, k = 1 is the identity function for some bit i, k=2 is x_i XOR x_j etc...


    Returns -
    TT - TruthTable object representing the function.
    k - indices of the bits that were chosen for the parity function.
    """
    TT = TruthTable(n, 1, is_random=False)
    inputs = TT.get_inputs_mat(n)
    bits_for_parity = np.random.choice(n, k, replace=False)
    y = np.bitwise_xor.reduce(inputs[:, bits_for_parity], axis=1)

    for input_idx in range(2**TT.domain_size):
        TT.set_f_x(input_idx, int(y[input_idx]))

    return TT, bits_for_parity

def get_random_k_sparse_AND(n, k):
    TT = TruthTable(n, 1, is_random=False)
    inputs = TT.get_inputs_mat(n)
    bits_for_AND = np.random.choice(n, k, replace=False)

    y = np.bitwise_and.reduce(inputs[:, bits_for_AND], axis=1)

    for input_idx in range(2**TT.domain_size):
        TT.set_f_x(input_idx, int(y[input_idx]))

    return TT, bits_for_AND

def powerset(iterable):
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))

def funcs_into_equivalence_classes(n, output_path=None):
    """
    Iterate over the entire 2^2^n space of dim=n boolean functions, and
    divide them into equivalence classes (where all functions in a classare permutations of each other)
    """
    if n > 4:
        raise ValueError(f"n > 4 classes cannot be calculated. (received n={n})")

    perms = list(itertools.permutations(range(n)))
    equivalence_classes_indices = []
    equivalence_classes = []

    for i, TT in enumerate(get_all_truth_tables(n)):
        print(f"Working on TT # {i}")


        if len(equivalence_classes) == 0:
            equivalence_classes.append(TT)
            equivalence_classes_indices.append([i])
            continue

        found_match = False

        TT_hamming_weight = TT.table.sum()

        for j, equiv_class in enumerate(equivalence_classes):
            equiv_class = equivalence_classes[j]
            representative_TT = equiv_class

            if TT_hamming_weight != representative_TT.table.sum():
                continue

            for perm in perms:
                permed_TT = TT.permute_truth_table_of_a_network(perm, permute_only_input=True)

                if int(permed_TT.get_similarity(representative_TT)) == 1:
                    print(f"\t\tFound a match!")
                    sys.stdout.flush()
                    sys.stderr.flush()
                    equivalence_classes_indices[j].append(i)
                    found_match = True
                    print(f"\t\tAdding TT # {i} to class {j}")
                    break

            if found_match == True:
                break

        # Open a new equiv. class
        if not found_match:
            equivalence_classes.append(TT)
            equivalence_classes_indices.append([i])
            print(f"\t\tCreating a new class for TT # {i}. Total classes = {len(equivalence_classes)}")


    print("#### DONE")
    print(f"Found {len(equivalence_classes)} classes for n={n}!")
    print("#### ")

    if output_path is not None:
        print(f"Saving the equivalence classes to {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(equivalence_classes_indices, f)
        print("#### DONE")

    return equivalence_classes_indices

def TT_to_fourier_expansion(TT):
    """
    A good reference for this is https://math.uchicago.edu/~may/REU2016/REUPapers/Spiro.pdf (equation 2.5)
    """
    flattened_TT_output = [1 if x == 1 else -1 for x in TT.table]

    possible_bit_indices = list(range(TT.domain_size))
    powerset_of_bit_indices = powerset(possible_bit_indices)

    all_inputs = list(itertools.product([-1, 1], repeat=TT.domain_size))

    fourier_expansion = []

    for subset in powerset_of_bit_indices:
        coeff = 0
        # This is basically the inner product part. TODO - move to a separate function?
        for i, inp in enumerate(all_inputs):
            f_value = flattened_TT_output[i]
            X_S = 1
            for bit_idx in subset:
                X_S *= inp[bit_idx]
            coeff += f_value * X_S

        coeff /= 2**TT.domain_size

        fourier_expansion.append((subset, coeff))
    return fourier_expansion

def get_max_degree_in_fourier_expansion(fourier_expansion):
    """
    Get the maximum degree with nonzero coefficients
    """
    max_degree = 0
    for subset, coeff in fourier_expansion:
        if len(subset) > max_degree and coeff != 0:
            max_degree = len(subset)
    return max_degree

def get_most_concentrated_degree_in_fourier_expansion(fourier_expansion):
    """
    Find the degree with the highest coefficient in the Fourier expansion, which means
    that the function is most concentrated at that degree.
    """
    # give slight edge for higher degrees so that it serves as a tie-breaker
    epsilon = 1e-6
    max_entry = max(fourier_expansion, key=lambda x: abs(x[1]) + epsilon*len(x[0]))
    max_degree = len(max_entry[0])

    return max_degree

def get_coeff_of_highest_xor(fourier_expansion):
    return abs(fourier_expansion[-1][-1])

def get_coeff_of_smallest_xor(fourier_expansion):
    return abs(fourier_expansion[0][-1])

def calculate_total_influence_for_TT(TT):
    influence_per_bit = [0 for x in range(TT.domain_size)]
    for i in range(TT.domain_size):
        for j in range(TT.num_rows):
            x = j
            x_flipped = x ^ (1 << (TT.domain_size - 1 - i))

            y = TT.get_f_x(x)
            y_flipped = TT.get_f_x(x_flipped)

            influence_per_bit[i] += abs(y_flipped[0] - y[0])

        influence_per_bit[i] /= TT.num_rows

    return sum(influence_per_bit)

def calculate_num_of_zero_inf_bits(TT):
    influence_per_bit = [0 for x in range(TT.domain_size)]
    for i in range(TT.domain_size):
        for j in range(TT.num_rows):
            x = j
            x_flipped = x ^ (1 << (TT.domain_size - 1 - i))

            y = TT.get_f_x(x)
            y_flipped = TT.get_f_x(x_flipped)

            influence_per_bit[i] += abs(y_flipped[0] - y[0])

        influence_per_bit[i] /= TT.num_rows

    return influence_per_bit.count(0)
