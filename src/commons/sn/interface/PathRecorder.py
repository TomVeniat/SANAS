import abc
import copy

import torch


class PathRecorder(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model):
        self.graph = model.graph
        self.default_out = model.out_node

        self.n_nodes = self.graph.number_of_nodes()

        # create node-to-index and index-to-node mapping
        self.node_index = {}
        self.rev_node_index = [None] * self.n_nodes
        for i, node in enumerate(model.traversal_order):
            self.node_index[node] = i
            self.rev_node_index[i] = node

        self.global_sampling = None
        self.n_samplings = 0

        self.active_nodes_seq = None
        self.samplings = None

        model.subscribe(self.new_event)

    def new_event(self, e):
        if e.type is 'new_sequence':
            self.new_sequence()
        elif e.type is 'new_iteration':
            self.new_iteration()
        elif e.type is 'sampling':
            self.new_sampling(e.node, e.value)

    def new_sequence(self):
        self.active_nodes_seq = []
        self.samplings = []

    def new_iteration(self):
        # Todo: adapt the global sampling idea in the sequence settings, maybe with a different global sampling for each class.
        # if self.default_out is not None and self.active is not None:
        #     pruned = self.get_pruned_architecture(self.default_out)
        #     self.update_global_sampling(pruned)

        self.active_nodes_seq.append(torch.Tensor())
        self.samplings.append(torch.Tensor())

    def new_sampling(self, node_name, sampling):
        """

        :param node_name: Noded considered
        :param sampling: Vector of size (batch_size), corresponding to the sampling of the given node
        :return:
        """
        if isinstance(sampling, torch.Tensor):
            sampling = sampling.cpu().squeeze()

        if sampling.dim() == 0:
            sampling.unsqueeze_(0)
        if sampling.dim() != 1:
            raise ValueError("'sampling' param should be of dimension one.")

        self.active = self.active_nodes_seq[-1]
        self.sampling = self.samplings[-1]

        batch_size = sampling.size(0)

        if self.active.numel() == 0 and self.sampling.numel() == 0:
            # This is the first step of the sequence
            self.active.resize_(self.n_nodes, self.n_nodes, batch_size).zero_()
            self.sampling.resize_(self.n_nodes, batch_size).zero_()

        node_ind = self.node_index[node_name]
        self.sampling[node_ind] = sampling

        # incoming is a (n_nodes*batch_size) matrix.
        # We will set incoming_{i,j} = 1 if the node i contributes to current node computation in batch element j
        incoming = self.active[node_ind]
        assert incoming.sum() == 0

        predecessors = list(self.graph.predecessors(node_name))

        if len(predecessors) == 0:
            # Considered node is the input node
            incoming[node_ind] += sampling

        for prev in predecessors:
            # If the predecessor itself is active (has connection with the input),
            # it could contribute to the computation of the considered node.
            incoming += self.active[self.node_index[prev]]

        assert incoming.size() == torch.Size((self.n_nodes, batch_size))

        # has_inputs[i] > 0 if there is at least one predecessor node which is active in batch element i
        has_inputs = incoming.max(0)[0]

        # the current node has outputs if it has at least on predecessor node active AND it is sampled
        has_outputs = ((has_inputs * sampling) != 0).float()

        backup = copy.deepcopy(incoming)

        ###
        # other_method = copy.deepcopy(incoming)
        # other_method[node_ind] += has_outputs
        # other_method = (other_method != 0).float()
        ###
        incoming[node_ind] += sampling

        sampling_mask = has_outputs.expand(self.n_nodes, batch_size)
        incoming *= sampling_mask

        res = (incoming != 0).float()
        self.active[node_ind] = res

        # eq = res.equal(other_method)
        # print(eq)

    def update_global_sampling(self, used_nodes):
        self.n_samplings += 1
        mean_sampling = used_nodes.mean(1).squeeze()

        if self.global_sampling is None:
            self.global_sampling = mean_sampling
        else:
            self.global_sampling += (1 / self.n_samplings) * (mean_sampling - self.global_sampling)

    def get_used_nodes(self, architectures):
        """
        Translates each architecture from a vector representation to a list of the nodes it contains
        :param architectures: a batch of architectures in format batch_size * n_nodes
        :return: a list of batch_size elements, each elements being a list of nodes.
        """
        res = []
        for arch in architectures:
            nodes = [self.rev_node_index[idx] for idx, used in enumerate(arch) if used == 1]
            res.append(nodes)
        return res

    def get_graph_paths(self, out_node):
        sampled, pruned = self.get_architectures(out_node)

        real_paths = []
        for i in range(pruned.size(1)):  # for each batch element
            path = [self.rev_node_index[ind] for ind, used in enumerate(pruned[:, i]) if used == 1]
            real_paths.append(path)

        res = self.get_used_nodes(pruned.t())

        assert real_paths == res

        sampling_paths = []
        for i in range(sampled.size(1)):  # for each batch element
            path = dict((self.rev_node_index[ind], elt) for ind, elt in enumerate(sampled[:, i]))
            sampling_paths.append(path)

        self.update_global_sampling(pruned)

        return real_paths, sampling_paths

    def get_posterior_weights(self):
        return dict((self.rev_node_index[ind], elt) for ind, elt in enumerate(self.global_sampling))

    def get_consistence(self, node):
        """
        Get an indicator of consistence for each sampled architecture up to the given node in last batch.

        :param node: The target node.
        :return: a ByteTensor containing one(zero) only if the architecture is consistent and the param is True(False).
        """
        return self.active[self.node_index[node]].sum(0) != 0

    def is_consistent(self, model):
        model.eval()
        with torch.no_grad():
            input = torch.ones(1, *model.input_size)
            model(input)
        consistence = self.get_consistence(model.out_node)
        return consistence.sum() != 0

    def get_architectures(self, out_nodes=None):
        if out_nodes is None:
            out_nodes = [self.default_out]
        return self.get_sampled_architectures(), self.get_pruned_architecture(out_nodes)

    def get_sampled_architectures(self):
        """

        :return: the real samplings in size (seq_len*n_nodes*batch_size)
        """
        return torch.stack(self.samplings)

    def get_pruned_architecture(self, out_nodes):
        """
        :return: the pruned samplings in size (seq_len*n_nodes*batch_size)
        """
        seq_len = len(self.active_nodes_seq)
        n_nodes = self.n_nodes
        batch_size = self.active_nodes_seq[0].size(-1)
        res = torch.zeros((seq_len, n_nodes, batch_size))
        for out_node in out_nodes:
            out_index = self.node_index[out_node]
            res += torch.stack([active[out_index] for active in self.active_nodes_seq])

        return (res!=0).float()

    def get_state(self):
        return {'node_index': self.node_index,
                'rev_node_index': self.rev_node_index,
                'global_sampling': self.global_sampling,
                'n_samplings': self.n_samplings}

    def load_state(self, state):
        for key, val in state.items():
            if not hasattr(self, key):
                raise AttributeError('Given state has unknown attribute {}.')
            setattr(self, key, val)
