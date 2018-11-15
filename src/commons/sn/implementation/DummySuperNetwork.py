import torch

from torch import nn
import numpy as np

from src.commons.sn.interface.NetworkBlock import DummyBlock
from src.commons.sn.networks.StochasticSuperNetwork import StochasticSuperNetwork


class DummySuperNetwork(StochasticSuperNetwork):
    n_comp_steps = 1
    NODE_NAME = 'ModuleNode'

    def __init__(self, input_dim, n_classes, static_node_proba, module, *args, **kwargs):
        super(DummySuperNetwork, self).__init__(*args, **kwargs)

        self._input_size = input_dim

        self.n_features = n_classes
        self.out_dim = n_classes

        self.static_node_proba = static_node_proba

        self.graph.add_node('ModuleNode', module=DummyBlock(module), sampling_param=0)

        self.sampling_parameters.append(nn.Parameter(torch.Tensor([np.inf]), requires_grad=False))
        self.blocks.append(module)

        self.feature_node = self.NODE_NAME

        self.set_graph(self.graph, self.NODE_NAME, self.NODE_NAME)
