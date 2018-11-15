from ..interface.CostEvaluator import CostEvaluator


class EdgeCostEvaluator(CostEvaluator):

    def get_cost(self, architectures):
        """

        :param architectures: a batch of architectures of size (n_nodes*batch_size)
        :return: a tensor of size (batch_size)
        """
        costs = self.costs.unsqueeze(1)
        costs = architectures * costs
        return costs.sum(0)

    @property
    def total_cost(self):
        return self.costs.sum()
