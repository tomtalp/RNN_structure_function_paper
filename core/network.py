import numpy as np
from copy import deepcopy
from core.truth_table import TruthTable
from typing import Iterable


class Network:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def shifted_sigmoid_half_net(x):
        return Network.sigmoid(x - (x.size - 1) / 2)

    @staticmethod
    def shifted_sigmoid_quarter_net(x):
        return Network.sigmoid(x - (x.size - 1) / 4)

    @staticmethod
    def shifted_sigmoid_fifth_net(x):
        return Network.sigmoid(x - (x.size - 1) / 5)

    def __init__(self, topology, initial_state=None, non_linearity=None):
        self._topology = deepcopy(topology)
        self._num_neurons = self._topology.shape[0]
        if initial_state is not None:
            self._state = deepcopy(initial_state.reshape((self._num_neurons, 1)))
        else:
            self._state = np.zeros((self._num_neurons, 1))
        if non_linearity is not None:
            self._non_linearity = non_linearity
        else:
            self._non_linearity = Network.shifted_sigmoid_half_net

        self._raw_truth_table = None
        self._binarized_truth_table = None

    def update_state(self, external_stimulus: np.ndarray):
        external_stimulus = external_stimulus.reshape((self._num_neurons, 1))
        self._state = np.round(self._non_linearity(self._topology @ self._state + external_stimulus))

    def calc_output(self, active_nodes):
        active_nodes = active_nodes.reshape((self._num_neurons, 1))
        return np.round(self._non_linearity(self._topology @ active_nodes))

    def get_state(self):
        return self._state

    def get_size(self):
        return self._num_neurons

    def get_topology(self):
        return self._topology

    def permute_network(self, reorder):
        reorder = reorder.reshape(self._num_neurons, 1).astype(int)
        new_topology = self._topology[reorder, reorder.T]
        new_state = self._state[reorder.flatten()]
        return Network(new_topology, new_state, self._non_linearity)

    def calc_raw_truth_table(self):
        inputs_mat = TruthTable.get_inputs_mat(self._num_neurons)
        self._raw_truth_table = inputs_mat @ self._topology.T

    def get_raw_truth_table(self):
        if self._raw_truth_table is None:
            self.calc_raw_truth_table()
        return self._raw_truth_table

    def calc_binarized_truth_table(self):
        inputs_mat = TruthTable.get_inputs_mat(self._num_neurons)
        prod = (inputs_mat @ self._topology.T).astype(float)
        for i in range(prod.shape[0]):
            prod[i] = np.round(self._non_linearity(prod[i]))
        self._binarized_truth_table = prod.astype(int)

    def get_binarized_truth_table(self) -> TruthTable:
        if self._binarized_truth_table is None:
            self.calc_binarized_truth_table()
        return TruthTable(self._num_neurons, self._num_neurons,
                          table=self._binarized_truth_table)

    def get_permuted_truth_table(self, permutation: Iterable[int]) -> TruthTable:
        permutation = np.array(permutation).astype(int).reshape(self._num_neurons, )
        if np.unique(
                permutation).size != self._num_neurons or permutation.min() != 0 or permutation.max() != self._num_neurons - 1:
            raise ValueError("the permutation is invalid")
        permutation_mat = np.eye(self._num_neurons)[permutation].astype(int)
        permed_topology = permutation_mat @ self._topology @ permutation_mat.T
        permed_net = Network(permed_topology, non_linearity=self._non_linearity)
        permuted_truth_table = permed_net.get_binarized_truth_table()
        return permuted_truth_table

    def get_raster(self, external_stim_over_time, do_update_state=False):
        if external_stim_over_time.shape[0] != self._num_neurons or external_stim_over_time.ndim != 2:
            raise ValueError("The dimension of the external stimulus must be #neurons X #time_steps")
        initial_state = self._state
        num_steps = external_stim_over_time.shape[1]
        raster = np.zeros((self._num_neurons, num_steps))
        for i in range(num_steps):
            self.update_state(external_stim_over_time[:, i])
            raster[:, i] = np.reshape(self._state, (self._num_neurons,))
        if not do_update_state:
            self._state = initial_state
        return raster.astype(int)
