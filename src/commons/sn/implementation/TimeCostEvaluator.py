import logging

import torch
from implementation.EdgeCostEvaluator import EdgeCostEvaluator
from interface.NetworkBlock import NetworkBlock

logger = logging.getLogger(__name__)


class TimeCostEvaluator(EdgeCostEvaluator):
    def __init__(self, bw, *args, **kwargs):
        super(TimeCostEvaluator, self).__init__(*args, **kwargs)
        self.bw = bw

    def init_costs(self, model, main_cost):
        model.eval()
        input = (torch.ones(1, *model.input_size),)
        graph = model.net

        self.costs = torch.Tensor(graph.number_of_nodes())

        graph.node[model.in_node]['input'] = [*input]
        for node in model.traversal_order:
            cur_node = graph.node[node]
            input = model.format_input(cur_node['input'])

            cur_mod = cur_node['module']
            out = cur_mod(input)

            if isinstance(cur_node['module'], NetworkBlock):
                cost = cur_node['module'].get_exec_time(input, self.bw)
            else:
                raise RuntimeError

            logger.info('Cost for {:10} at {:25}: {:.3e}ms '.format(type(cur_mod).__name__, str(node), cost))
            if main_cost:
                cur_node['cost'] = cost
            self.costs[self.path_recorder.node_index[node]] = cost
            cur_node['input'] = []

            for succ in graph.successors(node):
                if 'input' not in graph.node[succ]:
                    graph.node[succ]['input'] = []
                graph.node[succ]['input'].append(out)
