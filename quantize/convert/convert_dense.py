#-*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2019 hey-yahei
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import types
from collections import namedtuple

from mxnet.gluon.nn import Dense

from .ste_func import LinearQuantizeSTE

__all__ = ['gen_dense_converter']
__author__ = 'YaHei'

QuantizedArgs = namedtuple("DenseQuantizedArgs", "in_signed in_width wt_width quantize_input quant_type")


def _dense_forward(self, F, x, weight, bias=None, input_max=None):
    if self.enable_quantize:
        # Quantize input
        if self.quantize_args.quantize_input:
            self.current_input_max = F.max(F.abs(x), axis=1).mean().asscalar()
            if self.quantize_input:
                max_ = input_max.asscalar() if self.quantize_input_offline else self.current_input_max
                if self.quantize_args.in_signed:
                    in_scale = max_ / (2 ** (self.quantize_args.in_width - 1) - 1)
                else:
                    in_scale = max_ / (2 ** self.quantize_args.in_width - 1)
                # x = (x.clip(0., max_) / (in_scale + 1e-10)).round() * in_scale
                x = LinearQuantizeSTE(in_scale, max_)(x)

        # Simulate quantization for weight
        if self.quantize_args.quant_type == 'channel':
            num = self._units
            max_ = weight.abs().reshape((num, -1)).max(axis=1)
            wt_scale = max_ / (2 ** (self.quantize_args.wt_width - 1) - 1)
            wt_scale = wt_scale.reshape((num, 1))
            # weight_q = (weight / (wt_scale + 1e-10)).round() * wt_scale
            weight_q = LinearQuantizeSTE(wt_scale)(weight)
        else:
            max_ = weight.abs().max()
            wt_scale = max_ / (2 ** (self.quantize_args.wt_width - 1) - 1)
            # weight_q = (weight / (wt_scale + 1e-10)).round() * wt_scale
            weight_q = LinearQuantizeSTE(wt_scale)(weight)
    else:
        weight_q = weight

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


def gen_dense_converter(weight_width=8, input_signed=False, input_width=8, quantize_input=True, quant_type='layer'):
    if quant_type == "group":
        quant_type = "channel"

    def _converter(m):
        assert isinstance(m, Dense)

        if quantize_input:
            _add_quantize_input_params(m)
        m.origin_forward = m.hybrid_forward
        m.hybrid_forward = types.MethodType(_dense_forward, m)
        m.quantize_args = QuantizedArgs(in_signed=input_signed, in_width=input_width, wt_width=weight_width,
                                        quantize_input=quantize_input, quant_type=quant_type)
        m.enable_quantize = True
        m.quantize_input = quantize_input
    return _converter
