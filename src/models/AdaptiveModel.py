import itertools

import torch
from torch import nn
from torch.nn import Parameter
from tqdm import tqdm


class DistribGenerator(nn.Module):
    def __init__(self, latent_dim, distrib_dim, *args, **kwargs):
        super(DistribGenerator, self).__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.distrib_dim = distrib_dim
        self.lin = nn.Linear(latent_dim, distrib_dim)

    def forward(self, hidden):
        return self.lin(hidden).sigmoid()

    def set_bias(self, bias):
        self.lin.bias = Parameter(torch.ones(self.distrib_dim) * bias)


class AdaptiveModel(nn.Module):
    def __init__(self, stochastic_model, rnn, static, arch_bias, *args, **kwargs):
        super(AdaptiveModel, self).__init__(*args, **kwargs)
        self.stochastic_model = stochastic_model
        self.out_nodes = [stochastic_model.out_node, stochastic_model.feature_node]

        self.static = static

        # self.register_buffer('init_state', torch.zeros(rnn.hidden_size))
        self.init_state = Parameter(torch.zeros(rnn.hidden_size))
        self.rnn = rnn
        self.probas_gen = DistribGenerator(rnn.hidden_size, len(stochastic_model.sampling_parameters))
        if arch_bias is not None:
            self.probas_gen.set_bias(arch_bias)

        # self.features_dim = stochastic_model.features_dim

    def get_param_groups(self):
        groups = {
            'arch': {'name': 'arch_params', 'params': itertools.chain(self.rnn.parameters(),
                                                                      self.probas_gen.parameters())},
            'pred': {'name': 'pred_params', 'params': self.stochastic_model.parameters()}
        }

        return groups

    def forward(self, x, batch_first):
        """

        :param x:
        :param batch_first:
        :return:
        """
        self.stochastic_model.fire(type='new_sequence')
        self.stochastic_model.batched_log_probas = []

        split_dim = 1 if batch_first else 0
        batch_size = x.size(0) if batch_first else x.size(1)
        y_hat_list = []

        z_prev = self.init_state.expand(batch_size, self.rnn.hidden_size)

        self.probabilities = torch.tensor([])
        for x_t in tqdm(x.split(1, dim=split_dim), disable=batch_size > 1):
            x_t = x_t.squeeze(split_dim)
            y_pred, features = self.forward_sample(x_t, state=z_prev)
            y_hat_list.append(y_pred)
            z_t = self.rnn(features.view(features.size(0), -1), z_prev)
            z_prev = z_t

        return torch.stack(y_hat_list, dim=split_dim)

    def forward_sample(self, input, state):
        if self.static:
            # Do not generate the distribution dynamically
            probas = torch.ones(state.size(0), self.probas_gen.distrib_dim).to(state.device)
        else:
            probas = self.probas_gen(state)

        self.probabilities = torch.cat([self.probabilities, probas.unsqueeze(0).detach().cpu()])
        self.stochastic_model.sample_archs(probas)
        out, features = self.stochastic_model(input, return_features=True)
        return out, features
