import logging

import torch

from ..implementation.EdgeCostEvaluator import EdgeCostEvaluator

logger = logging.getLogger(__name__)


class SimpleEdgeCostEvaluator(EdgeCostEvaluator):

    def init_costs(self, model):
        graph = model.net
        self.costs = torch.ones(graph.number_of_nodes())
