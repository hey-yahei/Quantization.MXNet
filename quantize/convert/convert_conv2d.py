#-*- coding: utf-8 -*-

import types

from mxnet import nd
from mxnet.gluon.nn import Conv2D

__all__ = ['gen_conv2d_converter']
__author__ = "YaHei"


def _conv2d_forward(self, F, x, weight, bias=None,
                    weight_min=None, weight_max=None, input_min=None, input_max=None,
                    gamma=None, beta=None, running_mean=None, running_var=None):
    # Quantize input
    if input_min is not None:
        self.current_input_min = F.min(x).asscalar()
        self.current_input_max = F.max(x).asscalar()
        min = input_min.asscalar() if self.quantize_input_offline else self.current_input_min
        max = input_max.asscalar() if self.quantize_input_offline else self.current_input_max
        input_scale = (max - min) / (2 ** 8 - 1)
        x = F.round((F.clip(x, min, max) - min) / input_scale) * input_scale + min

    # Fake bn
    if gamma is not None:
        w_shape = weight.shape
        cout = w_shape[0]
        weight = (weight.reshape(cout, -1) * gamma.reshape(-1, 1) / F.sqrt(running_var + 1e-5).reshape(-1, 1)).reshape(w_shape)
        bias = gamma * (bias - running_mean) / F.sqrt(running_var + 1e-5) + beta

    # Simulate quantization for weight
    w_scale = (weight_max - weight_min) / (2 ** 8 - 1)
    weight_q = F.round((F.clip(weight, weight_min.asscalar(), weight_max.asscalar()) - weight_min) / w_scale) * w_scale + weight_min

    # Normal convolution
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
    if m.bias is None:
        m._kwargs['no_bias'] = False
        m.bias = m.params.get('bias',
                              shape=m.weight.shape[0],
                              init="zeros",
                              allow_deferred_init=True)


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


def gen_conv2d_converter(quantize_input=True, fake_bn=True):
    def _converter(m):
        assert isinstance(m, Conv2D)
        _add_quantize_weight_params(m)
        if quantize_input:
            _add_quantize_input_params(m)
        if fake_bn:
            _add_fake_bn_params(m)
            _add_fake_bn_ema_hook(m)
        m.hybrid_forward = types.MethodType(_conv2d_forward, m)
    return _converter
