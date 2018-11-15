import numpy as np
import torch
from torch import nn

from ..interface.NetworkBlock import ConvBn, Upsamp_Block, Add_Block, NetworkBlock, DummyBlock
from ..networks.StochasticSuperNetwork import StochasticSuperNetwork


def downsampling_layer(n_chan, k_size, bias=True, in_chan=None, size=None, rounding='ceil'):
    """
    :param size: the *input* size given to this layer
    :param rounding: The rounding method used to calculate the size of the downscaled feature maps.

    :return: A ConvBn layer with a stride of two.
     For a given square image a size N:
        if adapt is set to True (default), the output will be of size N/2 if N is even, (N+1)/2 if N is odd.
        if adapt is set to False, the output will be of size N/2 if N is even, (N-1)/2 if N is odd.
    """
    if rounding == 'ceil':
        padding = (1, 1)
    elif rounding == 'floor' and size:
        # we want a padding of 1 in the dim is Even, 0 if it's odd
        padding = tuple((1 - (dim_size % 2) for dim_size in size))
    else:
        raise RuntimeError

    in_chan = in_chan or n_chan
    return ConvBn(in_chan, n_chan, relu=False, k_size=k_size, stride=2, padding=padding, bias=bias)


def samesampling_layer(n_chan, k_size, bias=True, in_chan=None, ):
    in_chan = in_chan or n_chan
    return ConvBn(in_chan, n_chan, relu=False, k_size=k_size, bias=bias)


def upsampling_layer(n_chan, k_size, bias=True, size=None, in_chan=None):
    """
    :param size: the wanted *output* size for this layer.
    """
    in_chan = in_chan or n_chan
    return Upsamp_Block(in_chan, n_chan, False, k_size, bias, scale_size=size)


def in_module_factory(in_chan, n_chan, k_size, bias):
    return ConvBn(in_chan, n_chan, relu=False, k_size=k_size, bias=bias)


def out_module_factory(in_features, out_dim, bias):
    return Out(in_features, out_dim, bias)


class Out(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_features, out_dim, bias=True):
        super(Out, self).__init__()
        self.lin = nn.Linear(in_features, out_dim, bias)

    def forward(self, x):
        if isinstance(x, list):
            x = sum(x)
        x = x.view(x.size(0), -1)
        return self.lin(x)

    def get_flop_cost(self, x):
        y = self(x)
        return x[0].size(1) * y.size(1)


def get_scales(in_dim, downscale_rounding, n_scale):
    scales = [tuple(in_dim[-2:])]
    if not n_scale or n_scale < 0:
        n_scale = np.inf

    s = 1
    while 1 not in scales[-1] and s < n_scale:
        s += 1
        scales.append((int(downscale_rounding(scales[-1][0] / 2)), int(downscale_rounding(scales[-1][1] / 2))))
    return scales


class ThreeDimNeuralFabric(StochasticSuperNetwork):
    def __init__(self, n_layer, n_block, n_chan, input_dim, n_classes, static_node_proba, kernel_size=3, bias=True,
                 n_scale=0, rounding_method='ceil', adapt_first=False, *args, **kwargs):
        """
        Represents a 3 Dimensional Neural fabric, in which each layer, scale position has several identical blocks.
        :param n_layer:
        :param n_block:
        :param n_chan:
        :param data_prop:
        :param static_node_proba:
        :param kernel_size:
        :param bias:
        :param args:
        :param kwargs:
        """
        super(ThreeDimNeuralFabric, self).__init__(*args, **kwargs)
        self.static_node_proba = static_node_proba

        self.n_layer = n_layer
        self.n_block = n_block
        self.n_chan = n_chan
        self.kernel_size = kernel_size
        self.bias = bias

        self.adapt_first = adapt_first
        if rounding_method == 'ceil':
            downscale_rounding = np.ceil
        elif rounding_method == 'floor':
            downscale_rounding = np.floor
        else:
            raise ValueError("'downscale_rounding' param must be 'ceil' or 'floor' (got {})".format(rounding_method))
        self.downscale_rounding = downscale_rounding

        self._input_size = input_dim

        self.scales = get_scales(self.input_size, self.downscale_rounding, n_scale)
        self.n_scales = len(self.scales)

        self.out_size = n_classes

        self.loss = nn.CrossEntropyLoss(reduce=False)

        conv_params = (self.n_chan, self.kernel_size, self.bias)
        self.downsampling = lambda **kwargs: downsampling_layer(*conv_params, **kwargs, rounding=rounding_method)
        self.samesampling = lambda **kwargs: samesampling_layer(*conv_params, **kwargs)
        self.upsampling = lambda **kwargs: upsampling_layer(*conv_params, **kwargs)

        for i in range(1, n_layer):
            self._add_layer(i)

        self._connect_input()
        self._connect_output()

        self.set_graph(self.graph, 'In', 'Out')

    def _add_layer(self, layer_idx):
        """
        Add a layer to the network's graph. A layer spans on the two-dimensional (scale, block) axes.
        This method will create and connect nodes to the (layer_idx-1) layer, following the connectivity defined by
        _get_scales_connections and _get_blocks_connections.
        :param layer_idx: the depth of the layer to add.
        """
        for scale in range(self.n_scales):
            input_scales = self._get_scales_connections(scale)
            self._add_block(layer_idx - 1, input_scales, layer_idx, scale)

    def _add_block(self, input_layer, input_scales, blk_layer, blk_scale, skip_agg=False):
        for block in range(self.n_block):
            self._add_node(input_layer, input_scales, blk_layer, blk_scale, block, skip_agg=skip_agg)

    def _add_node(self, in_layer, in_scales, layer, scale, block, skip_agg=False):
        transformations = []
        in_blocks = self._get_blocks_connections(block)
        for s in in_scales:
            for b in in_blocks:
                tr = self._add_transformation(in_layer, s, b, layer, scale, block)
                transformations.append(tr)

        self._add_aggregation(layer, scale, block, transformations, skip_agg)

    def _add_transformation(self, s_l, s_s, s_b, d_l, d_s, d_b):
        if not self.adapt_first and s_l == 0 and s_s == 0:
            in_scale = self._input_size[0]
        else:
            in_scale = None

        if s_s < d_s:
            module = self.downsampling(size=self.scales[s_s], in_chan=in_scale)
        elif s_s == d_s:
            module = self.samesampling(in_chan=in_scale)
        else:
            module = self.upsampling(size=self.scales[d_s], in_chan=in_scale)

        cur_node = ((s_l, s_s, s_b), (d_l, d_s, d_b))

        sampling_param = self.sampling_param_generator()

        self.graph.add_node(cur_node, module=module, sampling_param=len(self.sampling_parameters))

        self.graph.add_edge(cur_node[0], cur_node, width_node=cur_node)

        self.sampling_parameters.append(sampling_param)
        self.blocks.append(module)
        return cur_node

    def _add_aggregation(self, layer, scale, block, inputs, skip_agg):
        cur_node = (layer, scale, block)

        for i in inputs:
            self.graph.add_edge(i, cur_node, width_node=cur_node)

        if skip_agg:
            return

        module = Add_Block()
        sampling_param = self.sampling_param_generator(static=True)

        self.graph.add_node(cur_node, module=module, sampling_param=len(self.sampling_parameters))

        # for i in inputs:
        #     self.graph.add_edge(i, cur_node, width_node=cur_node)

        self.blocks.append(module)
        self.sampling_parameters.append(sampling_param)

    def _connect_input(self):

        in_block = 'In'
        sampling_param = self.sampling_param_generator(static=True)
        if self.adapt_first:
            mod = in_module_factory(self._input_size[0], self.n_chan, self.kernel_size, self.bias)
        else:
            mod = DummyBlock()

        self.graph.add_node(in_block, module=mod, sampling_param=len(self.sampling_parameters))

        self.sampling_parameters.append(sampling_param)
        self.blocks.append(mod)

        for block in range(self.n_block):
            # Connect all the blocks in first scale, first layer to the Input block
            cur_node = (0, 0, block)
            self.graph.add_edge(in_block, cur_node, width_node=cur_node)

        for scale in range(self.n_scales):
            input_scales = self._get_scales_connections(scale, is_zip=True)
            self._add_block(0, input_scales, 0, scale)

    def _connect_output(self):
        out_block = 'Out'
        sampling_param = self.sampling_param_generator(static=True)

        self.n_features = self.n_chan * self.scales[-1][0] * self.scales[-1][1]
        mod = out_module_factory(self.n_features, self.out_size, self.bias)

        self.graph.add_node(out_block, module=mod, sampling_param=len(self.sampling_parameters))

        self.sampling_parameters.append(sampling_param)
        self.blocks.append(mod)

        for block in range(self.n_block):
            # Connect all the blocks in last scale, last layer to the Out block
            cur_node = (self.n_layer - 1, self.n_scales - 1, block)
            self.graph.add_edge(cur_node, out_block, width_node=out_block)

        if self.n_layer > 1:
            for scale in range(self.n_scales):
                input_scales = self._get_scales_connections(scale, is_zip=True)
                self._add_block(self.n_layer - 1, input_scales, self.n_layer - 1, scale, skip_agg=True)

    def sampling_param_generator(self, static=False):
        trainable = self.static_node_proba < 0

        if static:
            trainable = False
            param_value = np.inf
        elif trainable:
            param_value = self.INIT_NODE_PARAM
        else:
            param_value = np.inf if np.random.rand() < self.static_node_proba else -np.inf

        return nn.Parameter(torch.Tensor([param_value]), requires_grad=trainable)

    def _get_scales_connections(self, cur_scale, is_zip=False):
        """
        Defines the scale dimension connectivity, default to [cur_scale-1, cur_scale, cur_scale+1].
        :param cur_scale:
        :return:
        """
        min_scale = np.max([0, cur_scale - 1])

        if is_zip:
            max_scale = cur_scale
        else:
            max_scale = np.min([self.n_scales, cur_scale + 2])
        return range(min_scale, max_scale)

    def _get_blocks_connections(self, cur_block):
        """
        Defines the block dimension connectivity, default to all blocks.
        """
        return range(self.n_block)

    def __str__(self):
        return super(ThreeDimNeuralFabric, self).__str__() + "\tScales: {}".format(self.scales)
