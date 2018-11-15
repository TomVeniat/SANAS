import logging

import torch

from ..implementation.EdgeCostEvaluator import EdgeCostEvaluator
from ..interface.NetworkBlock import NetworkBlock

logger = logging.getLogger(__name__)


class ComputationCostEvaluator(EdgeCostEvaluator):

    def init_costs(self, model):
        with torch.no_grad():
            input_var = (torch.ones(1, *model.input_size),)
            graph = model.net

            self.costs = torch.Tensor(graph.number_of_nodes())

            graph.node[model.in_node]['input'] = [*input_var]
            for node in model.traversal_order:
                cur_node = graph.node[node]
                input_var = model.format_input(cur_node['input'])

                out = cur_node['module'](input_var)

                if isinstance(cur_node['module'], NetworkBlock):
                    cost = cur_node['module'].get_flop_cost(input_var)
                else:
                    raise RuntimeError

                self.costs[self.node_index[node]] = cost
                cur_node['input'] = []

                for succ in graph.successors(node):
                    if 'input' not in graph.node[succ]:
                        graph.node[succ]['input'] = []
                    graph.node[succ]['input'].append(out)
