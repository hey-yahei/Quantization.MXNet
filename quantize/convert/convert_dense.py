#-*- coding: utf-8 -*-

import types
from collections import namedtuple

from mxnet.gluon.nn import Dense

__all__ = ['gen_dense_converter']
__author__ = 'YaHei'

QuantizedArgs = namedtuple("DenseQuantizedArgs", "in_signed in_width wt_width quantize_input")


def _dense_forward(self, F, x, weight, bias=None, input_max=None):
    # Quantize input
    if self.quantize_args.quantize_input:
        self.current_input_max = F.max(F.abs(x)).asscalar()
        max_ = input_max.asscalar() if self.quantize_input_offline else self.current_input_max
        if self.quantize_args.in_signed:
            scale = max_ / (2 ** (self.quantize_args.in_width - 1) - 1)
        else:
            scale = max_ / (2 ** self.quantize_args.in_width - 1)
        x = (x.clip(0., max_) / (scale + 1e-10)).round() * scale

    # Simulate quantization for weight
    max_ = weight.abs().max()
    scale = max_ / (2 ** (self.quantize_args.wt_width - 1) - 1)
    weight_q = (weight / (scale + 1e-10)).round() * scale

    # Normal dense
    act = self.origin_forward(F, x, weight_q, bias)

    return act


def _add_quantize_input_params(m):
    m.quantize_input_offline = False
    m.current_input_max = 0.
    m.input_max = m.params.get("input_max",
                               shape=(1,), init="zeros",
                               allow_deferred_init=True,
                               differentiable=False)


def gen_dense_converter(weight_width=8, input_signed=False, input_width=8, quantize_input=True):
    def _converter(m):
        assert isinstance(m, Dense)

        if quantize_input:
            _add_quantize_input_params(m)
        m.origin_forward = m.hybrid_forward
        m.hybrid_forward = types.MethodType(_dense_forward, m)
        m.quantize_args = QuantizedArgs(in_signed=input_signed, in_width=input_width, wt_width=weight_width, quantize_input=quantize_input)
    return _converter
