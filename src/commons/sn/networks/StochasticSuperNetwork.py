import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from ..interface.Observable import Observable
from ..networks.SuperNetwork import SuperNetwork


class StochasticSuperNetwork(Observable, SuperNetwork):
    INIT_NODE_PARAM = 3

    def __init__(self, deter_eval, *args, **kwargs):
        super(StochasticSuperNetwork, self).__init__(*args, **kwargs)
        self.samp_params = nn.ParameterList()

        self.blocks = nn.ModuleList([])
        self.graph = nx.DiGraph()

        self.nodes_param = None
        self.probas = None
        self.mean_entropy = None
        self.deter_eval = deter_eval
        self.all_same = False

        self.batched_sampling = None
        self.batched_log_probas = None

    def get_sampling(self, node_name, out):
        """
        Get a batch of sampling for the given node on the given output.
        Fires a "sampling" event with the node name and the sampling Variable.
        :param node_name: Name of the node to sample
        :param out: Tensor on which the sampling will be applied
        :return: A Variable brodcastable to out size, with all dimensions equals to one except the first one (batch)
        """

        batch_size = out.size(0)
        sampling_dim = [batch_size] + [1] * (out.dim() - 1)

        node = self.net.node[node_name]
        if self.all_same:
            sampling = self.batched_sampling[:, node['sampling_param']].expand(sampling_dim)
        else:
            sampling = self.batched_sampling[:, node['sampling_param']].contiguous().view(sampling_dim)

        return sampling

    def forward(self, *input, return_features=False):
        if len(input) != 1:
            raise RuntimeError("SSN forward's input must be a single tensor, got {}".format(len(input)))
        feats = None

        self.net.node[self.in_node]['input'] = [*input]
        for node in self.traversal_order:
            # globals()['__builtins__']['input'](f'layer {node}')
            cur_node = self.net.node[node]
            input = self.format_input(cur_node.pop('input'))

            if len(input) == 0:
                raise RuntimeError('Node {} has no inputs'.format(node))

            out = cur_node['module'](input)
            if isinstance(out, tuple):
                out, feats = out

            sampling = self.get_sampling(node, out)
            out = out * sampling

            if node == self.out_node:
                feats = input if feats is None else feats
                return out, feats if return_features else out

            for succ in self.net.successors(node):
                if 'input' not in self.net.node[succ]:
                    self.net.node[succ]['input'] = []
                self.net.node[succ]['input'].append(out)

    def sample_archs(self, probas=None, all_same=False):
        """
        :param probas: B_size*N_nodes Tensor containing the probability of each arch being sampled.
        :param all_same: if True, the same sampling will be used for the whole batch in the next forward.
        :return:
        """
        if probas.dim() != 2 or all_same and probas.size(0) != 1:
            raise ValueError('probas params has wrong dimension: {} (all_same={})'.format(probas.size(), all_same))
        self._sample_archs(probas)
        self.all_same = all_same

        self._fire_all_samplings()

    def _sample_archs(self, probas):
        distrib = torch.distributions.Bernoulli(probas)

        if not self.training and self.deter_eval:
            self.batched_sampling = (probas > 0.5).float()
        else:
            self.batched_sampling = distrib.sample()

        self.batched_log_probas.append(distrib.log_prob(self.batched_sampling))

    def _fire_all_samplings(self):
        """
        Method used to notify the observers of the sampling
        """
        self.fire(type='new_iteration')

        for node_name in self.traversal_order:
            # todo: Implemented this way to work with old implementation, can be done in a better way now.
            node = self.net.node[node_name]
            sampling = self.batched_sampling[:, node['sampling_param']]
            self.fire(type='sampling', node=node_name, value=sampling)

    @property
    def sampling_parameters(self):
        print('/!\ Getting samp params /!\ ')
        return self.samp_params

    @property
    def n_layers(self):
        return sum([mod.n_layers for mod in self.blocks])

    @property
    def n_comp_steps(self):
        return sum([mod.n_comp_steps for mod in self.blocks])

    @property
    def ordered_node_names(self):
        return [str(elt[0]) for elt in sorted(self.net.nodes.data('sampling_param'), key=lambda x: x[1])]

    def update_probas_and_entropies(self):
        if self.nodes_param is None:
            self._init_nodes_param()
        self.probas = {}
        self.entropies = {}
        self.mean_entropy = .0
        for node, props in self.graph.nodes.items():
            param = self.sampling_parameters[props['sampling_param']]
            p = param.sigmoid().item()
            self.probas[node] = p
            if p in [0, 1]:
                e = 0
            else:
                e = -(p * np.log2(p)) - ((1 - p) * np.log2(1 - p))
            self.entropies[node] = e
            self.mean_entropy += e
        self.mean_entropy /= self.graph.number_of_nodes()

    def _init_nodes_param(self):
        self.nodes_param = {}
        for node, props in self.graph.node.items():
            if 'sampling_param' in props and props['sampling_param'] is not None:
                self.nodes_param[node] = props['sampling_param']

    def __str__(self):
        model_descr = 'Model:{}\n' \
                      '\t{} nodes\n' \
                      '\t{} blocks\n' \
                      '\t{} parametrized layers\n' \
                      '\t{} computation steps\n' \
                      '\t{} parameters ({} trainable)\n' \
                      '\t{} meta-params\n'
        return model_descr.format(type(self).__name__, self.graph.number_of_nodes(), len(self.blocks), self.n_layers,
                                  self.n_comp_steps, sum(i.numel() for i in self.parameters()),
                                  sum(i.numel() for i in self.parameters() if i.requires_grad),
                                  len(self.sampling_parameters))
