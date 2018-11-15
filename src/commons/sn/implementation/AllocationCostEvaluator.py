import networkx as nx
import torch

from interface.CostEvaluator import CostEvaluator


class AllocationCostEvaluator(CostEvaluator):

    def __init__(self, n_para, *args, **kwargs):
        super(AllocationCostEvaluator, self).__init__(*args, **kwargs)
        self.n_parallel = n_para
        self.graph = self.path_recorder.graph
        self.node_steps = None
        self.cache = {}

    @property
    def total_cost(self):
        self.node_steps = []

        return self._get_cost(list(nx.topological_sort(self.graph)), self.n_parallel)[0]

    def get_costs(self, architectures):
        assert len(architectures) == 2
        sampled_arch, pruned_arch = architectures
        pruned_cost = self.get_cost(pruned_arch)
        return [pruned_cost, pruned_cost]

    def get_cost(self, arch):
        used_path = self.path_recorder.get_used_nodes(arch.t())
        res = []
        self.node_steps = []
        for path in used_path:
            # path = self.filter_path(path)
            path_key = frozenset(path)
            if path_key not in self.cache:
                self.cache[path_key] = self._get_cost(path, self.n_parallel)

            res.append(self.cache[path_key][0])
            self.node_steps.append(self.cache[path_key][1])
        return torch.Tensor(res)

    def _get_cost(self, path, n_parallel):
        n_steps = 0
        computing_nodes = []
        computing_sum = 0
        # nodes_step = defaultdict(int)
        nodes_step = {}
        while len(path) > 0:
            i = 0
            # while len(computing_nodes) < n_parallel and i < len(path):
            while computing_sum < n_parallel and i < len(path):
                node = path[i]

                if self.is_computable(node, path):
                    computing_nodes.append(node)
                    nodes_step[node] = n_steps
                    computing_sum += self.graph.node[node]['module'].n_comp_steps
                #     print('OK\t: {}'.format(node))
                # else:
                #     print('NOK\t: {}'.format(node))

                i += 1

            path = [elt for elt in path if elt not in computing_nodes]
            # print('Step {} - Computing={} ({}/{})'.format(n_steps,computing_nodes, computing_sum, n_parallel))

            if computing_sum > 0:
                n_steps += 1
            computing_nodes = []
            computing_sum = 0

        return n_steps, nodes_step

    def is_computable(self, node, nodes_to_compute):
        for in_node in self.graph.predecessors(node):
            if in_node in nodes_to_compute:
                return False
        return True

    def filter_path(self, path):
        raise RuntimeError
        return [node for node in path if self.graph.node[node]['module'].n_comp_steps > 0]

    def new_epoch(self):
        self.cache = {}
