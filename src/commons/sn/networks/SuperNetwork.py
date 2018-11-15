import networkx as nx
from torch import nn


class SuperNetwork(nn.Module):
    def __init__(self):
        super(SuperNetwork, self).__init__()

        self.net = None
        self.traversal_order = None
        self.in_node = None
        self.out_node = None
        self.observer = None

    def set_graph(self, network, in_node, out_node):
        self.net = network
        if not nx.is_directed_acyclic_graph(self.net):
            raise ValueError('A Super Network must be defined with a directed acyclic graph')

        self.traversal_order = list(nx.topological_sort(self.net))
        self.in_node = in_node
        self.out_node = out_node

        # TODO Allow several input and/or output nodes
        if self.traversal_order[0] != in_node or self.traversal_order[-1] != out_node:
            raise ValueError('Seems like the given graph is broken')

    def forward(self, *input):
        output = []
        self.net.node[self.in_node]['input'] = [*input]

        for node in self.traversal_order:
            cur_node = self.net.node[node]
            input = self.format_input(cur_node['input'])
            out = cur_node['module'](input)
            cur_node['input'] = []

            if node == self.out_node:
                output.append(out)

            for succ in self.net.successors_iter(node):
                if 'input' not in self.net.node[succ]:
                    self.net.node[succ]['input'] = []
                self.net.node[succ]['input'].append(out)

        return output[0]

    @property
    def input_size(self):
        if not hasattr(self, '_input_size'):
            raise RuntimeError('SuperNetworks should have an `_input_size` attribute.')
        return self._input_size

    def get_ops_per_node(self):
        return dict((node_name, node_data['module'].get_flop_cost())
                    for node_name, node_data in dict(self.graph.nodes(True)).items())

    def get_params_per_node(self):
        return dict((node_name, sum(param.numel() for param in node_data['module'].parameters()))
                    for node_name, node_data in dict(self.graph.nodes(True)).items())

    @staticmethod
    def format_input(input):
        if (isinstance(input, tuple) or isinstance(input, list)) and len(input) == 1:
            input = input[0]
        return input
