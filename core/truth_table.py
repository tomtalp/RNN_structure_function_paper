import numpy as np
from torch.utils.data import Dataset
import torch


class TruthTable:
    @staticmethod
    def get_inputs_mat(domain_size):
        rows = np.arange(2 ** domain_size)
        shifts = np.arange(domain_size - 1, -1, -1)
        return ((rows[:, None] >> shifts) & 1).astype(int)

    @staticmethod
    def _to_row_idx(x):
        if isinstance(x, (int, np.integer)):
            return int(x)
        bits = np.asarray(x).astype(int).flatten()
        return int(bits.dot(1 << np.arange(bits.size - 1, -1, -1)))

    def __init__(self, domain_size, range_size, is_random=False, table=None):
        self.domain_size = domain_size
        self.row_len = range_size
        self.num_rows = 2 ** self.domain_size
        if table is not None:
            table = np.asarray(table)
            if table.shape != (self.num_rows, self.row_len):
                raise ValueError("The dimensions of the provided table and the domain and range sizes don't match")
            self.table = table.copy()
        elif is_random:
            self.table = np.random.randint(0, 2, size=(self.num_rows, self.row_len))
        else:
            self.table = np.zeros((self.num_rows, self.row_len), dtype=int)

    def get_f_x(self, x):
        return self.table[self._to_row_idx(x)]

    def set_f_x(self, x, y):
        self.table[self._to_row_idx(x)] = y

    def get_self_inputs_mat(self):
        return TruthTable.get_inputs_mat(self.domain_size)

    def calc_output_dist(self):
        dist = np.zeros(2 ** self.row_len)
        for row in range(self.num_rows):
            row_int = TruthTable._to_row_idx(self.table[row])
            dist[row_int] += 1
        dist /= self.num_rows
        return dist

    def calc_outputs_hamming_weights_dist(self):
        hamming_weights_dist = np.zeros(self.row_len + 1)
        for row in range(self.num_rows):
            hamming_weights_dist[self.table[row].sum()] += 1
        hamming_weights_dist /= self.num_rows
        return hamming_weights_dist

    def permute_truth_table_of_a_network(self, permutation, permute_only_input=False):
        permutation = np.asarray(list(permutation.values()) if isinstance(permutation, dict) else permutation)
        if not np.array_equal(np.sort(permutation), np.arange(len(permutation))):
            raise ValueError("invalid permutation")
        permuted_tt = TruthTable(self.domain_size, self.row_len)
        inputs_mat = TruthTable.get_inputs_mat(self.domain_size)
        for r in range(self.num_rows):
            out_word = self.table[r]
            permed_in_word = inputs_mat[r][permutation]
            permed_out_word = out_word[permutation] if not permute_only_input else out_word
            permuted_tt.set_f_x(permed_in_word, permed_out_word)
        return permuted_tt

    def get_similarity(self, other):
        if not isinstance(other, TruthTable):
            raise TypeError("TruthTables can be compared only to TruthTables")
        if self.domain_size != other.domain_size or self.row_len != other.row_len:
            raise ValueError("the sizes of the truth tables don't match")
        return (self.table == other.table).sum() / self.num_rows

    def __str__(self):
        inputs = TruthTable.get_inputs_mat(self.domain_size)
        output_str = ""
        for x, y in zip(inputs, self.table):
            x_str = ", ".join(str(bit) for bit in x)
            y_str = ", ".join(str(bit) for bit in y)
            output_str += f"f({x_str}) = {y_str}\n"
        return output_str

    def __eq__(self, other):
        if not isinstance(other, TruthTable):
            raise TypeError("TruthTables can be compared only to TruthTables")
        if self.domain_size != other.domain_size or self.row_len != other.row_len:
            raise ValueError("the sizes of the truth tables don't match")
        return np.array_equal(self.table, other.table)


class TruthTableDataset(Dataset):
    def __init__(self, truth_table):
        self.truth_table = truth_table

    def __len__(self):
        return self.truth_table.num_rows

    def __getitem__(self, idx):
        X = TruthTable.get_inputs_mat(self.truth_table.domain_size)[idx]
        y = self.truth_table.get_f_x(idx)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
