#-*- coding: utf-8 -*-

import types
from collections import namedtuple

from mxnet import nd
from mxnet.gluon.nn import Conv2D

__all__ = ['gen_conv2d_converter']
__author__ = "YaHei"


QuantizedArgs = namedtuple("ConvQuantizedArgs", "in_width wt_width quantize_input fake_bn quant_type")

def _conv2d_forward(self, F, x, weight, bias=None, input_max=None,
                    gamma=None, beta=None, running_mean=None, running_var=None):
    # Quantize input
    if self.quantize_args.quantize_input:
        self.current_input_max = F.max(F.abs(x)).asscalar()
        max_ = input_max.asscalar() if self.quantize_input_offline else self.current_input_max
        scale = max_ / (2 ** self.quantize_args.in_width - 1)
        x = (x.clip(0., max_) / scale).round() * scale

    # Fake bn
    if self.quantize_args.fake_bn:
        w_shape = weight.shape
        cout = w_shape[0]
        weight = (weight.reshape(cout, -1) * gamma.reshape(-1, 1) / F.sqrt(running_var + 1e-5).reshape(-1, 1)).reshape(w_shape)
        bias = gamma * (bias - running_mean) / F.sqrt(running_var + 1e-5) + beta

    # Simulate quantization for weight
    if self.quantize_args.quant_type == 'channel':
        num = self._kwargs['num_filter']
        max_ = weight.abs().reshape((num, -1)).max(axis=1)
        scale = max_ / (2 ** (self.quantize_args.wt_width - 1) - 1)
        scale = scale.reshape((num, 1, 1, 1))
        weight_q = (weight / scale).round() * scale
    elif self.quantize_args.quant_type == 'group':
        num = self._kwargs['num_group']
        max_ = weight.abs().reshape((num, -1)).max(axis=1)
        scale = max_ / (2 ** (self.quantize_args.wt_width - 1) - 1)
        scale = scale.reshape((num, 1, 1, 1))
        weight_q = (weight / scale).round() * scale
    else:
        max_ = weight.abs().max()
        scale = max_ / (2 ** (self.quantize_args.wt_width - 1) - 1)
        weight_q = (weight / scale).round() * scale

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
        m.current_mean = ( y.sum(axis=(2,3)).sum(axis=0) ) / num_samples
        diff_square = (y - m.current_mean.reshape(1,-1,1,1)) ** 2
        m.current_var = ( diff_square.sum(axis=(2,3)).sum(axis=0) ) / num_samples
    m.register_forward_pre_hook(_ema_hook)

def gen_conv2d_converter(weight_width=8, input_width=8, quantize_input=False, fake_bn=False, quant_type="layer"):
    def _converter(m):
        assert isinstance(m, Conv2D)

        if quantize_input:
            _add_quantize_input_params(m)
        if fake_bn:
            _add_fake_bn_params(m)
            _add_fake_bn_ema_hook(m)
        m.origin_forward = m.hybrid_forward
        m.hybrid_forward = types.MethodType(_conv2d_forward, m)
        m.quantize_args = QuantizedArgs(in_width=input_width, wt_width=weight_width,
                                        quantize_input=quantize_input, fake_bn=fake_bn, quant_type=quant_type)
    return _converter