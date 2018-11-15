import abc

import torch


class CostEvaluator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, node_index, *args, **kwargs):
        super(CostEvaluator, self).__init__()
        self.node_index = node_index
        self.costs = None

    @abc.abstractmethod
    def get_cost(self, **kwargs):
        raise NotImplementedError

    def get_costs(self, architectures):
        """
        :param architectures:  a tensor of size (seq_len*n_nodes*batch_size)
        :return: a tensor of size (seq_len*batch_size)
        """
        return torch.stack([self.get_cost(arch.squeeze(0)) for arch in architectures.split(1, 0)])

    def init_costs(self, *args, **kwargs):
        pass

    def get_state(self):
        return {'costs': self.costs}

    def load_state(self, state):
        for key, val in state.items():
            assert hasattr(self, key)
            setattr(self, key, val)

    def new_epoch(self):
        pass
