#-*- coding: utf-8 -*-

import types

from mxnet import nd
from mxnet.gluon.nn import Conv2D

__all__ = ["convert_conv2d", "convert_conv2d_quantize_input"]
__author__ = "YaHei"


def _conv2d_forward(self, F, x, weight, bias=None,
                    weight_min=None, weight_max=None):
    w_scale = (weight_max - weight_min) / (2 ** 8 - 1)
    weight_q = F.round((F.clip(weight, weight_min.asscalar(), weight_max.asscalar()) - weight_min) / w_scale) * w_scale + weight_min

    if bias is None:
        act = getattr(F, self._op_name)(x, weight_q, name='fwd', **self._kwargs)
    else:
        act = getattr(F, self._op_name)(x, weight_q, bias, name='fwd', **self._kwargs)

    if self.act is not None:
        act = self.act(act)

    return act


def _clip_per_row(data, mins, maxs):
    rows = [nd.clip(x, min, max).expand_dims(0) for x in data for min in mins for max in maxs]
    return nd.concat(*rows)


def _conv2d_forward_quantize_input(self, F, x, weight, bias=None,
                                   weight_min=None, weight_max=None, input_min=None, input_max=None):
    self.current_input_min = F.min(x).asscalar()
    self.current_input_max = F.max(x).asscalar()
    min = input_min.asscalar() if self.quantize_input_offline else self.current_input_min
    max = input_max.asscalar() if self.quantize_input_offline else self.current_input_max
    input_scale = (max - min) / (2 ** 8 - 1)
    x = F.round((F.clip(x, min, max) - min) / input_scale) * input_scale + min

    w_scale = (weight_max - weight_min) / (2 ** 8 - 1)
    weight_q = F.round((F.clip(weight, weight_min.asscalar(), weight_max.asscalar()) - weight_min) / w_scale) * w_scale + weight_min

    if bias is None:
        act = getattr(F, self._op_name)(x, weight_q, name='fwd', **self._kwargs)
    else:
        act = getattr(F, self._op_name)(x, weight_q, bias, name='fwd', **self._kwargs)

    if self.act is not None:
        act = self.act(act)

    return act


def _add_quantize_weight_params(m):
    m.weight_min = m.params.get('weight_min',
                                shape=(1,), init="zeros",
                                allow_deferred_init=True)
    m.weight_max = m.params.get('weight_max',
                                shape=(1,), init="ones",
                                allow_deferred_init=True)


def _add_quantize_input_params(m):
    m.quantize_input_offline = False
    m.current_input_min = 0.
    m.current_input_max = 0.
    m.input_min = m.params.get("input_min",
                               shape=(1,), init="zeros",
                               allow_deferred_init=True,
                               differentiable=False)
    m.input_max = m.params.get("input_max",
                               shape=(1,), init="zeros",
                               allow_deferred_init=True,
                               differentiable=False)


def convert_conv2d(m):
    assert isinstance(m, Conv2D)
    _add_quantize_weight_params(m)
    m.hybrid_forward = types.MethodType(_conv2d_forward, m)


def convert_conv2d_quantize_input(m):
    assert isinstance(m, Conv2D)
    _add_quantize_weight_params(m)
    _add_quantize_input_params(m)
    m.hybrid_forward = types.MethodType(_conv2d_forward_quantize_input, m)
