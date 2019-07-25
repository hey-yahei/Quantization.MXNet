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

from mxnet import nd
from mxnet.gluon.nn import Conv2D

from .ste_func import LinearQuantizeSTE

__all__ = ['gen_conv2d_converter']
__author__ = "YaHei"


QuantizedArgs = namedtuple("ConvQuantizedArgs",
                           "quantize_input in_signed in_width "
                           "wt_width quant_type "
                           "fake_bn")


def _conv2d_forward(self, F, x, weight, bias=None, input_max=None,
                    gamma=None, beta=None, running_mean=None, running_var=None):
    # Fake bn
    if self.quantize_args.fake_bn:
        w_shape = weight.shape
        cout = w_shape[0]
        weight = (weight.reshape(cout, -1) * gamma.reshape(-1, 1) / F.sqrt(running_var + 1e-10).reshape(-1, 1)).reshape(w_shape)
        bias = gamma * (bias - running_mean) / F.sqrt(running_var + 1e-10) + beta

    if self.enable_quantize:
        # Quantize input
        if self.quantize_args.quantize_input:
            self.current_input_max = F.max(F.abs(x), axis=(1, 2, 3)).mean().asscalar()
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
            num = self._kwargs['num_filter']
            max_ = weight.abs().reshape((num, -1)).max(axis=1)
            wt_scale = max_ / (2 ** (self.quantize_args.wt_width - 1) - 1)
            wt_scale = wt_scale.reshape((num, 1, 1, 1))
            # weight_q = (weight / (wt_scale + 1e-10)).round() * wt_scale
            weight_q = LinearQuantizeSTE(wt_scale)(weight)
        elif self.quantize_args.quant_type == 'group':
            num = self._kwargs['num_group']
            max_ = weight.abs().reshape((num, -1)).max(axis=1)
            wt_scale = max_ / (2 ** (self.quantize_args.wt_width - 1) - 1)
            wt_scale = wt_scale.reshape((num, 1, 1, 1))
            # weight_q = (weight / (wt_scale + 1e-10)).round() * wt_scale
            weight_q = LinearQuantizeSTE(wt_scale)(weight)
        else:
            max_ = weight.abs().max()
            wt_scale = max_ / (2 ** (self.quantize_args.wt_width - 1) - 1)
            # weight_q = (weight / (wt_scale + 1e-10)).round() * wt_scale
            weight_q = LinearQuantizeSTE(wt_scale)(weight)
    else:
        weight_q = weight

    # Normal convolution
    act = self.origin_forward(F, x, weight_q, bias)

    return act


def _add_quantize_input_params(m):
    m.quantize_input_offline = False
    m.current_input_max = 0.
    m.input_max = m.params.get("input_max",
                               shape=(1,), init="zeros",
                               allow_deferred_init=True,
                               differentiable=False)


def _add_fake_bn_params(m):
    in_channels = m._kwargs['num_filter']
    m.gamma = m.params.get('gamma',
                           shape=(in_channels,), init="ones",
                           allow_deferred_init=True,
                           differentiable=True)
    m.beta = m.params.get('beta',
                          shape=(in_channels,), init="zeros",
                          allow_deferred_init=True,
                          differentiable=True)
    m.running_mean = m.params.get('running_mean',
                                  shape=(in_channels,),
                                  init="zeros",
                                  allow_deferred_init=True,
                                  differentiable=False)
    m.running_var = m.params.get('running_var',
                                 shape=(in_channels,),
                                 init="ones",
                                 allow_deferred_init=True,
                                 differentiable=False)


def _add_fake_bn_ema_hook(m):
    def _ema_hook(m, x):
        x = x[0]
        weight = m.weight.data()
        bias = nd.zeros(shape=weight.shape[0], ctx=weight.context) if m.bias is None else m.bias.data()
        y = nd.Convolution(x, weight, bias, **m._kwargs)
        num_samples = y.shape[0] * y.shape[2] * y.shape[3]
        m.current_mean = y.sum(axis=(0, 2, 3)) / num_samples
        diff_square = (y - m.current_mean.reshape(1, -1, 1, 1)) ** 2
        m.current_var = diff_square.sum(axis=(0, 2, 3)) / num_samples
    m.register_forward_pre_hook(_ema_hook)


def gen_conv2d_converter(weight_width=8, quant_type="layer",
                         quantize_input=True, input_signed=False, input_width=8,
                         fake_bn=False):
    def _converter(m):
        assert isinstance(m, Conv2D)

        if quantize_input:
            _add_quantize_input_params(m)
        if fake_bn:
            _add_fake_bn_params(m)
            _add_fake_bn_ema_hook(m)
        m.origin_forward = m.hybrid_forward
        m.hybrid_forward = types.MethodType(_conv2d_forward, m)
        m.quantize_args = QuantizedArgs(in_signed=input_signed, in_width=input_width, wt_width=weight_width,
                                        quantize_input=quantize_input, fake_bn=fake_bn, quant_type=quant_type)
        m.enable_quantize = True
        m.quantize_input = quantize_input
    return _converter
