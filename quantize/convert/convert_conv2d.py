#-*- coding: utf-8 -*-

import types

from mxnet import nd
from mxnet.gluon.nn import Conv2D

__all__ = ['gen_conv2d_converter']
__author__ = "YaHei"


# def _quantize_simulate(data, signed, width, one_side=False, min=None, max=None, training=True):
#     # online
#     if max is None:
#         if signed:
#             max = data.abs().max()
#             min = -max
#         else:
#             max = data.max()
#             min = data.min()
#     # get scalar
#     if type(max) == nd.NDArray:
#         max_s = max.asscalar()
#         if not all(not signed, one_side):
#             min = min_s = 0.
#         else:
#             min_s = min.asscalar()
#     # not training
#     if not training:
#         max = max_s
#         min = min_s
#     # quantize
#     scalar = (max - min) / (2 ** width - 1)
#     if signed:
#         res = (data.clip(min_s, max_s) / scalar).round() * scalar
#     else:
#         res = ((data.clip(min_s, max_s) - min) / scalar).round() * scalar + min
#     return res


def _conv2d_forward(self, F, x, weight, bias=None,
                    weight_min=None, weight_max=None, input_min=None, input_max=None,
                    gamma=None, beta=None, running_mean=None, running_var=None):
    # Quantize input
    if input_max is not None:
        if self.quantize_kwargs['input']['signed']:
            self.current_input_max = F.max(F.abs(x)).asscalar()
            self.current_input_min = -self.current_input_max
        else:
            self.current_input_max = F.max(x).asscalar()
            self.current_input_min = F.min(x).asscalar() if not self.quantize_kwargs['input']['one_side'] else 0.
        max = input_max.asscalar() if self.quantize_input_offline else self.current_input_max
        if not self.quantize_kwargs['input']['signed'] and self.quantize_kwargs['input']['one_side']:
            min = 0.
        else:
            min = input_min.asscalar() if self.quantize_input_offline else self.current_input_min
        if self.quantize_kwargs['input']['signed']:
            input_scale = max / (2 ** (self.quantize_kwargs['input']['width'] - 1) - 1)
            x = F.round((F.clip(x, min, max)) / input_scale) * input_scale
        else:
            input_scale = (max - min) / (2 ** self.quantize_kwargs['input']['width'] - 1)
            x = F.round((F.clip(x, min, max) - min) / input_scale) * input_scale + min

    # Fake bn
    if gamma is not None:
        w_shape = weight.shape
        cout = w_shape[0]
        weight = (weight.reshape(cout, -1) * gamma.reshape(-1, 1) / F.sqrt(running_var + 1e-5).reshape(-1, 1)).reshape(w_shape)
        bias = gamma * (bias - running_mean) / F.sqrt(running_var + 1e-5) + beta

    # Simulate quantization for weight
    if not self.quantize_kwargs['weight']['training']:
        if self.quantize_kwargs['weight']['signed']:
            weight_max = F.max(F.abs(weight))
            weight_min = -weight_max
        else:
            weight_max = F.max(weight)
            weight_min = F.min(weight) if not self.quantize_kwargs['weight']['one_side'] else nd.array([0.])
    if self.quantize_kwargs['weight']['signed']:
        w_scale = weight_max / (2 ** (self.quantize_kwargs['weight']['width'] - 1) - 1)
        weight_q = F.round(F.clip(weight, weight_min.asscalar(), weight_max.asscalar()) / w_scale) * w_scale
    else:
        w_scale = (weight_max - weight_min) / (2 ** self.quantize_kwargs['weight']['width'] - 1)
        weight_q = F.round((F.clip(weight, weight_min.asscalar(), weight_max.asscalar()) - weight_min) / w_scale) * w_scale + weight_min

    # Normal convolution
    if bias is None:
        act = getattr(F, self._op_name)(x, weight_q, name='fwd', **self._kwargs)
    else:
        act = getattr(F, self._op_name)(x, weight_q, bias, name='fwd', **self._kwargs)
    if self.act is not None:
        act = self.act(act)

    return act


def _add_quantize_weight_params(m, one_side=False):
    m.weight_min = m.params.get('weight_min',
                                shape=(1,), init="zeros",
                                allow_deferred_init=True) if not one_side else None
    m.weight_max = m.params.get('weight_max',
                                shape=(1,), init="ones",
                                allow_deferred_init=True)


def _add_quantize_input_params(m, one_side=True):
    m.quantize_input_offline = False
    m.current_input_min = 0. if not one_side else None
    m.current_input_max = 0.
    m.input_min = m.params.get("input_min",
                               shape=(1,), init="zeros",
                               allow_deferred_init=True,
                               differentiable=False) if not one_side else None
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


def gen_conv2d_converter(quantize_input=True, fake_bn=False,
                         weight_signed=True, weight_width=8, weight_one_side=False, weight_training=True,
                         input_signed=False, input_width=8, input_one_side=True):
    def _converter(m):
        assert isinstance(m, Conv2D)
        if weight_training:
            _add_quantize_weight_params(m, weight_one_side and not weight_signed)
        if quantize_input:
            _add_quantize_input_params(m, input_one_side and not input_signed)
        if fake_bn:
            _add_fake_bn_params(m)
            _add_fake_bn_ema_hook(m)
        m.hybrid_forward = types.MethodType(_conv2d_forward, m)
        m.quantize_kwargs = {
            "input": {
                "signed": input_signed,
                "width": input_width,
                "one_side": input_one_side
            },
            "weight": {
                "signed": weight_signed,
                "width": weight_width,
                "one_side": weight_one_side,
                "training": weight_training
            }
        }
    return _converter
