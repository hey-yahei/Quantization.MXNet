#-*- coding: utf-8 -*-

import types

from mxnet import nd
from mxnet.initializer import Constant
from mxnet.gluon import nn
from mxnet.gluon.parameter import ParameterDict

__all__ = ['freeze_conv2d']


def _get_quantized_forward(m):
    def _conv2d_forward(self, F, x, weight, bias=None,
                        weight_min=None, weight_max=None, input_min=None, input_max=None):
        act = F.contrib.quantized_conv(x, weight, bias,
                                       min_data=input_min, max_data=input_max,
                                       min_weight=weight_min, max_weight=weight_max,
                                       name="fwd", **self._kwargs)
        return act if self.act is None else self.act(act)
    return types.MethodType(_conv2d_forward, m)


def _freeze_params(m):
    weight, min, max = nd.contrib.quantize(m.weight.data(), m.weight_min.data(), m.weight_max.data(), out_type="uint8")

    # m._params = ParameterDict()
    # m.weight_q = m.params.get("weight_quantize", shape=weight.shape,
    #                         init=Constant(weight), dtype="uint8")
    # m.bias_q = None
    #
    # m.weight_q.initialize()


def freeze_conv2d(m):
    assert isinstance(m, nn.Conv2D)
    m.hybrid_forward = _get_quantized_forward(m)
    _freeze_params(m)
