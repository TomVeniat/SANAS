import copy
from collections import OrderedDict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.commons.sn.interface.NetworkBlock import NetworkBlock, Add_Block, DummyBlock
from src.commons.sn.networks.StochasticSuperNetwork import StochasticSuperNetwork

MODELS = {
    'CNN_TRAD_POOL2': dict(k_w=(20, 10), k_h=(8, 4), h=(64, 64), p_w=(2, 1), p_h=(2, 1), s_w=(1, 1), s_h=(1, 1)),
    'CNN_TRAD_FPOOL3': dict(k_w=(20, 10), k_h=(8, 4), h=(64, 64, 32, 128), p_w=(1, 1), p_h=(3, 1), s_w=(1, 1),
                            s_h=(1, 1)),
    'CNN_ONE_FPOOL3': dict(k_w=(32,), k_h=(8,), h=(54, 32, 128, 128), p_w=(1,), p_h=(3,), s_w=(1,), s_h=(1,)),
    'CNN_ONE_FSTRIDE4': dict(k_w=(32,), k_h=(8,), h=(186, 32, 128, 128), p_w=(1,), p_h=(1,), s_w=(1,), s_h=(4,)),
    'CNN_ONE_FSTRIDE8': dict(k_w=(32,), k_h=(8,), h=(336, 32, 128, 127), p_w=(1,), p_h=(1,), s_w=(1,), s_h=(8,)),
    'CNN_ONE_TSTRIDE2': dict(k_w=(16, 9), k_h=(8, 4), h=(78, 78, 32), p_w=(1, 1), p_h=(3, 1), s_w=(2, 1), s_h=(1, 1)),
    'CNN_ONE_TSTRIDE4': dict(k_w=(16, 5), k_h=(8, 4), h=(100, 78, 32), p_w=(1, 1), p_h=(3, 1), s_w=(4, 1), s_h=(1, 1)),
    'CNN_ONE_TSTRIDE8': dict(k_w=(16, 5), k_h=(8, 4), h=(126, 78, 32), p_w=(1, 1), p_h=(3, 1), s_w=(8, 1), s_h=(1, 1))

}


class ConvPoolBlock(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_chan, conv_ksize, conv_stride, pool_ksize, relu, bias=True):
        super(ConvPoolBlock, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=conv_ksize, stride=conv_stride, bias=bias)
        self.pool = nn.MaxPool2d(kernel_size=pool_ksize)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        if self.relu:
            x = F.relu(x)
        return x

    def get_flop_cost(self, x):
        y = self.conv(x)
        return self.get_conv2d_flops(x, y, self.conv.kernel_size, self.conv.stride)


class LinearBlock(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_f, out_f, relu, bias=True, reshape_input=False):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(in_f, out_f, bias=bias)
        self.relu = relu
        self.reshape_input = reshape_input

    def forward(self, x):
        if self.reshape_input:
            x = x.view(x.size(0), -1)

        x = self.fc(x)
        if self.relu:
            x = F.relu(x)
        return x

    def get_flop_cost(self, x):
        return self.fc.in_features * self.fc.out_features


def kws_cnn_factory(in_size, n_classes, model='CNN_TRAD_POOL2', h_params=None, bias=True, full=False, sep_features=False, *args, **kwargs):
    """
    Given an input of size t*f features in time and frequency dimensions.
    :param in_size:
    :param n_classes:
    :param model:
    :param h_params: dict containing all hyper parameters (Ignored if model if specified):
        - k_w: filter widths, in the time domain.
        - k_h: filter heights, in the frequency domain.
        - h: number of output feature maps for conv layers, number of hidden units for fc layers (out fc wil be added).
        - p_w: Pooling in the time domain after each conv layer.
        - p_h: Pooling in the frequency domain after each conv layer.
        - s_w: Stride in time domain for each conv layer.
        - s_h: Stride in frequency domain for each conv layer.
    :param bias: Use bias in Conv an FC layers (default:True).
    :return:
    """
    if model:
        h_params = copy.deepcopy(MODELS[model])

    conv_layers = OrderedDict()
    last_out_size = in_size[1:]
    last_out_h = in_size[0]

    h_params['h'] = h_params['h'] + (n_classes,)

    linear_shortcuts = OrderedDict()
    short_idx = 0

    # Create all conv layers:
    for i, (m, r, n, p, q, s, v) in enumerate(zip(h_params['k_w'], h_params['k_h'], h_params['h'], h_params['p_w'],
                                                  h_params['p_h'], h_params['s_w'], h_params['s_h'])):
        cpb = ConvPoolBlock(last_out_h, n, conv_ksize=(m, r), conv_stride=(s, v), pool_ksize=(p, q), relu=True,
                            bias=bias)
        conv_layers['conv_{}'.format(i)] = cpb

        if full:
            n_features = last_out_h * last_out_size[0] * last_out_size[1]
            linear_shortcuts['shorcut_{}'.format(short_idx)] = dict(in_f=n_features, relu=True, reshape_input=True,
                                                                    bias=bias)
            short_idx += 1

        last_out_h = n
        last_out_size = int((last_out_size[0] - m) / s) + 1, int((last_out_size[1] - r) / v) + 1  # After Conv
        last_out_size = int((last_out_size[0] - p) / p) + 1, int((last_out_size[1] - q) / q) + 1  # After Pool

    n_features = last_out_h * last_out_size[0] * last_out_size[1]

    lin_layers = OrderedDict()
    for j, n in enumerate(h_params['h'][i + 1:]):
        lb = LinearBlock(n_features, n, relu=True, reshape_input=(j == 0), bias=bias)
        lin_layers['lin_{}'.format(j)] = lb

        if full:
            linear_shortcuts['shorcut_{}'.format(short_idx)] = dict(in_f=n_features, relu=True, reshape_input=(j == 0),
                                                                    bias=bias)
            short_idx += 1

        n_features = n

    # Remove last Linear shortcut
    if full:
        linear_shortcuts = list(linear_shortcuts.items())[:-1]

    linear_shortcuts = [(name, LinearBlock(out_f=h_params['h'][-2], **opts)) for name, opts in linear_shortcuts]

    return KWS_CNN(in_size, conv_layers, lin_layers, shortcut_lins=linear_shortcuts, features_dim=h_params['h'][-2],
                   sep_features=sep_features, *args, **kwargs)


class KWS_CNN(StochasticSuperNetwork):
    IN_NODE_NAME = 'In'
    OUT_FEATURES_NODE_NAME = 'out_feat'
    RNN_FEATURES_NODE_NAME = 'rnn_feat'

    def __init__(self, in_size, convs, linears, static_node_proba, shortcut_lins, features_dim, sep_features=False, *args, **kwargs):
        super(KWS_CNN, self).__init__(*args, **kwargs)
        assert isinstance(convs, OrderedDict) and isinstance(linears, OrderedDict)

        self._input_size = in_size
        self.features_dim = features_dim
        self.sep_features = sep_features

        self.static_node_proba = static_node_proba

        self.blocks = nn.ModuleList([])
        self.sampling_parameters = nn.ParameterList()
        self.graph = nx.DiGraph()

        # Create a list containing every modules except the last output
        all_modules = list(convs.items()) + list(linears.items())
        out = all_modules.pop()

        self.add_step(self.OUT_FEATURES_NODE_NAME, Add_Block(), x=len(all_modules))
        if sep_features:
            self.add_step(self.RNN_FEATURES_NODE_NAME, Add_Block(), x=len(all_modules))

        prev = self.add_step(self.IN_NODE_NAME, DummyBlock())

        for i, mod in enumerate(all_modules):
            self.add_step(mod[0], mod[1], prev)

            if shortcut_lins and i < len(shortcut_lins):
                self.add_shortcut(shortcut_lins[i][0], shortcut_lins[i][1], prev, self.OUT_FEATURES_NODE_NAME)

            prev = mod[0]

        self.graph.add_edge(prev, self.OUT_FEATURES_NODE_NAME, width_node=self.OUT_FEATURES_NODE_NAME)
        self.add_step(out[0], out[1], self.OUT_FEATURES_NODE_NAME)

        self.set_graph(self.graph, self.IN_NODE_NAME, out[0])

        self.feature_node = self.RNN_FEATURES_NODE_NAME if self.sep_features else self.OUT_FEATURES_NODE_NAME

    def add_step(self, name, module, source=None, x=None, y=None):
        sampling_param = self.sampling_param_generator()

        self.graph.add_node(name, module=module, sampling_param=len(self.sampling_parameters))
        if source:
            self.graph.add_edge(source, name, width_node=name)

        self.sampling_parameters.append(sampling_param)
        self.blocks.append(module)
        return name

    def add_shortcut(self, name, module, source, dest):
        sampling_param = self.sampling_param_generator()

        self.graph.add_node(name, module=module, sampling_param=len(self.sampling_parameters))
        self.graph.add_edge(source, name, width_node=name)
        self.graph.add_edge(name, dest, width_node=dest)

        self.sampling_parameters.append(sampling_param)
        self.blocks.append(module)

        if self.sep_features:
            sampling_param = self.sampling_param_generator()
            block = DummyBlock()
            dummy_name = 'feats_' + name
            self.graph.add_node(dummy_name, module=block, sampling_param=len(self.sampling_parameters))
            self.graph.add_edge(name, dummy_name, width_node=dummy_name)
            self.graph.add_edge(dummy_name, self.RNN_FEATURES_NODE_NAME, width_node=self.RNN_FEATURES_NODE_NAME)

            self.sampling_parameters.append(sampling_param)
            self.blocks.append(module)

    def sampling_param_generator(self):
        if self.static_node_proba >= 0:
            param_value = 1 if np.random.rand() < self.static_node_proba else -1
            param_value *= np.inf
            trainable = False
        else:
            param_value = self.INIT_NODE_PARAM
            trainable = True
        return nn.Parameter(torch.Tensor([param_value]), requires_grad=trainable)
