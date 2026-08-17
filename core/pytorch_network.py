import torch
import numpy as np
from core.network import Network
from core.truth_table import TruthTable
from utils.nn_models import pad_1d_tensor, epsilon_mask
from typing import Iterable


class PytorchNet_ReadoutNeuron(torch.nn.Module):
    """
    A Pytorch network with a connectivity matrix W, and a readout neuron.
    """
    def __init__(self, runtime, dim, activation_fn, W_bias=False, Wout_bias=False):
        super(PytorchNet_ReadoutNeuron, self).__init__()

        self.runtime = runtime
        self.dim = dim

        self.W = torch.nn.Parameter(torch.randn(self.dim, self.dim, dtype=torch.double), requires_grad=True)
        self.Wout = torch.nn.Parameter(torch.randn(self.dim, dtype=torch.double), requires_grad=True)

        self.W_bias_flag = W_bias
        self.W_bias = torch.nn.Parameter(torch.randn(self.dim, dtype=torch.double), requires_grad=True)

        self.Wout_bias_flag = Wout_bias
        self.Wout_bias = torch.nn.Parameter(torch.randn(1, dtype=torch.double), requires_grad=True)
        self.relu = torch.nn.ReLU()

        if activation_fn == "relu":
            self.activation = torch.nn.ReLU()
        elif activation_fn == "sigmoid":
            self.activation = torch.sigmoid

    def forward(self, x):
        for _ in range(self.runtime):
            x = torch.matmul(self.W, x)

            if self.W_bias_flag:
                x = x + self.W_bias

            x = self.activation(x)

        out = torch.matmul(x, self.Wout)

        if self.Wout_bias_flag:
            out = out + self.Wout_bias.squeeze(0)

        return out

class PytorchNet_Interneurons_And_Readout(torch.nn.Module):
    """
    A Pytorch network with a N neurons, out of which only a fraction is used to receive inputs.
    """
    def __init__(self, runtime, input_dim, internal_dim, activation_fn, W_bias=False, Wout_bias=False):
        super(PytorchNet_Interneurons_And_Readout, self).__init__()

        self.runtime = runtime
        self.input_dim = input_dim
        self.internal_dim = internal_dim

        self.W = torch.nn.Parameter(torch.randn(self.internal_dim, self.internal_dim, dtype=torch.double), requires_grad=True)
        self.Wout = torch.nn.Parameter(torch.randn(self.internal_dim, dtype=torch.double), requires_grad=True)

        self.W_bias_flag = W_bias
        self.W_bias = torch.nn.Parameter(torch.randn(self.internal_dim, dtype=torch.double), requires_grad=True)

        self.Wout_bias_flag = Wout_bias
        self.Wout_bias = torch.nn.Parameter(torch.randn(1, dtype=torch.double), requires_grad=True)

        if activation_fn == "relu":
            self.activation = torch.nn.ReLU()
        elif activation_fn == "sigmoid":
            self.activation = torch.sigmoid


    def forward(self, x):
        x_padded = pad_1d_tensor(x, self.internal_dim).double()
        for rt in range(self.runtime):
            x_padded = torch.matmul(self.W, x_padded)

            if self.W_bias_flag:
                x_padded = x_padded + self.W_bias

            x_padded = self.activation(x_padded)


        out = torch.matmul(x_padded, self.Wout)

        if self.Wout_bias_flag:
            out = out + self.Wout_bias.squeeze(0)

        return out

    def debug_forward(self, x):
        x_padded = pad_1d_tensor(x, self.internal_dim).double()
        print(f"Initial x = {x}. x_padded = {x_padded}")
        for rt in range(self.runtime):
            x_padded = torch.matmul(self.W, x_padded)

            if self.W_bias_flag:
                x_padded = x_padded + self.W_bias

            x_padded = torch.sigmoid(x_padded)
            print(f"After rt={rt} - {x_padded}")


        out = torch.matmul(x_padded, self.Wout)

        if self.Wout_bias_flag:
            out = out + self.Wout_bias.squeeze(0)
        print(f"\tOut before sigmoid = {out}")
        out = torch.sigmoid(out)
        print(f"\tOut AFTER sigmoid = {out}")
        return out

class PytorchNetwork(Network):
    def __init__(self, topology, pytorch_network_class, pytorch_network_args, initial_state=None, activation_fn=None, final_threshod=0, device=None):
        super(PytorchNetwork, self).__init__(topology, initial_state, activation_fn)

        self.device = device


        if pytorch_network_class == PytorchNet_ReadoutNeuron:
            self._input_size = self._num_neurons
        elif pytorch_network_class == PytorchNet_Interneurons_And_Readout:
            self._input_size = pytorch_network_args["input_dim"]
        else:
            raise NotImplementedError(f"Pytorch network of type {pytorch_network_class} isn't implemented!")

        self._inputs_mat = TruthTable.get_inputs_mat(self._input_size)

        self._threshold = final_threshod
        self._output_size = 1
        self.pytorch_network = pytorch_network_class(**pytorch_network_args)

        self.pytorch_network = self.pytorch_network.to(device)

        self._topology_as_tensor = torch.tensor(self._topology).to(self.device)

        self._mask_epsilon = 1e-12
        self.apply_mask()

    def apply_mask(self):
        new_W = self.pytorch_network.W.data * self._topology_as_tensor
        self.pytorch_network.W.data = new_W

    def get_pytorch_params(self):
        return self.pytorch_network.parameters()

    def _apply_threshold(self, x):
        if x < self._threshold:
            return 0
        return 1

    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            for (X, y) in dataloader:
                pass

    def permute_network(self, reorder):
        raise NotImplementedError()

    def calc_raw_truth_table(self):
        raise NotImplementedError()

    def get_raw_truth_table(self):
        raise NotImplementedError()

    def calc_binarized_truth_table(self):
        outputs = np.empty((len(self._inputs_mat), 1))
        for i, x in enumerate(self._inputs_mat):
            x = torch.tensor(x).double()
            x = x.to(self.device)
            y = self.__call__(x)
            y = self._apply_threshold(y)
            outputs[i] = y
        self._binarized_truth_table = outputs.astype(int)

    def get_binarized_truth_table(self) -> TruthTable:
        self.calc_binarized_truth_table()
        return TruthTable(self._input_size, self._output_size,
                          table=self._binarized_truth_table)

    def get_permuted_truth_table(self, permutation: Iterable[int]) -> TruthTable:
        raise NotImplementedError()

    def get_raster(self, external_stim_over_time: np.ndarray, do_update_state: bool = False):
        raise NotImplementedError()

    def __call__(self, x):
        return self.pytorch_network(x)


class SparseMaskedRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_steps, mask):
        super(SparseMaskedRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_steps = num_steps


        self.input_weights = torch.nn.Parameter(torch.randn(input_size) * 0.1)

        # Recurrent weights and mask
        self.recurrent_weights = torch.nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.recurrent_mask = mask
        self.register_buffer('r_mask', self.recurrent_mask)
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

        self.activation = torch.nn.Sigmoid()
        self.readout = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, input_size = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Input-to-hidden
        transformed_input = torch.mul(x, self.input_weights)
        if transformed_input.size(1) < self.hidden_size:
            pad_size = self.hidden_size - transformed_input.size(1)
            transformed_input = torch.nn.functional.pad(transformed_input, (0, pad_size), mode='constant', value=0)

        h_t += transformed_input

        # self.recurrent_weights = self.recurrent_weights.to(x.device)
        for _ in range(self.num_steps):
            masked_weights = self.recurrent_weights * self.recurrent_mask  # Apply mask on the same device
            h_t = self.activation(torch.matmul(h_t, masked_weights) + self.bias) # x @ W means W_ij is i --> j

        # Readout
        output = self.readout(h_t)
        return output

    def apply_masks(self):
        with torch.no_grad():
            self.recurrent_weights *= self.recurrent_mask
