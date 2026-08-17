import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.truth_table import TruthTable
from core.pytorch_network import PytorchNet_ReadoutNeuron, PytorchNetwork
from utils.topology import get_all_topologies
from utils.truth_tables import get_truth_table_by_idx
from utils.training import run_training_loop

DIM = 3
RUNTIME = 3
FUNC_INDICES_TO_TRY = [0, 23, 105, 255]  # constant-0, majority, parity, constant-1

device = torch.device("cpu")

for topology_idx, topology in enumerate(get_all_topologies(DIM)):
    if topology_idx > 4:  # only a handful of topologies for a quick demo run
        break
    if topology.sum() == 0:
        continue

    for func_idx in FUNC_INDICES_TO_TRY:
        TT = get_truth_table_by_idx(func_idx, DIM)
        net = PytorchNetwork(
            topology, PytorchNet_ReadoutNeuron,
            {"runtime": RUNTIME, "dim": DIM, "activation_fn": "sigmoid", "W_bias": True, "Wout_bias": True},
            device=device,
        )
        log = run_training_loop(net, TT, lr=0.1, epochs=300, device=device)
        print(f"topology_idx={topology_idx} func_idx={func_idx} final_acc={log['final_acc']:.3f}")
