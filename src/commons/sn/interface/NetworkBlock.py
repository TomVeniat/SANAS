import timeit

import torch
import torch.nn.functional as F
from torch import nn


class NetworkBlock(nn.Module):
    def get_exec_time(self, x, bw=False):
        """
        :param x: The Input tensor for the current NetworkBlock
        :param bw: Evaluate Time including backward if True (default:False).
        :return:
        """
        if bw:
            def func():
                if isinstance(x, torch.Tensor):
                    input = torch.tensor(x.detach(), requires_grad=True)
                else:
                    input = [torch.tensor(elt.detach(), requires_grad=True) for elt in x]
                out = self(input)
                out.backward(out.detach())
        else:
            def func():
                self(x)

        n_exec, time = timeit.Timer(func).autorange()
        mean_time = time * 1000 / n_exec
        return mean_time

    @staticmethod
    def get_conv2d_flops(x, y, k_size, s_size=(1, 1)):
        assert x.dim() == 4 and y.dim() == 4
        return x.size(1) * y.size(1) * y.size(2) * y.size(3) * k_size[0] * k_size[1] / (s_size[0] * s_size[1])


class DummyBlock(NetworkBlock):
    n_layers = 0
    n_comp_steps = 0

    def __init__(self, mod=None):
        super(DummyBlock, self).__init__()
        self.mod = mod

    def forward(self, x):
        if self.mod:
            return self.mod(x)
        return x

    def get_flop_cost(self, x):
        return 0


class ConvBn(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_chan, relu, k_size=3, stride=1, padding=1, dilatation=1, bias=True):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=k_size, stride=stride, padding=padding, dilation=dilatation, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x)
        return x

    def get_flop_cost(self, x):
        y = self(x)
        return self.get_conv2d_flops(x, y, self.conv.kernel_size)


class Upsamp_Block(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_chan, relu, k_size, bias, scale_size, scale_factor=2):
        super(Upsamp_Block, self).__init__()
        self.conv_layer = ConvBn(in_chan, out_chan, relu=relu, k_size=k_size, bias=bias)
        self.scale_factor = scale_factor
        self.scale_size = scale_size
        self.rescale_params = dict(mode='bilinear', align_corners=False)
        if self.scale_size:
            self.rescale_params['size'] = self.scale_size
        else:
            self.rescale_params['scale_factor'] = self.scale_factor

    def forward(self, x):
        x = F.interpolate(x, **self.rescale_params)
        return self.conv_layer(x)

    def get_flop_cost(self, x):
        return self.conv_layer.get_flop_cost(x)


class Add_Block(NetworkBlock):
    n_layers = 0
    n_comp_steps = 1

    def forward(self, x):
        if not isinstance(x, list):
            return F.relu(x)
        assert isinstance(x, list)
        return F.relu(sum(x))

    def get_flop_cost(self, x):
        if not isinstance(x, list):
            return 0
        assert isinstance(x, list)
        return x[0].numel() * (len(x) - 1)
