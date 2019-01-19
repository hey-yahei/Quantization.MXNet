#-*- coding: utf-8 -*-

import types

from mxnet import nd
from mxnet.gluon.parameter import ParameterDict
from mxnet.gluon import nn

__all__ = ['merge_bn']


def _bypass_bn(net, exclude=[]):
    def _forward(self, F, x, *args, **kwargs):
        return x
    def _bypass(m):
        if isinstance(m ,nn.BatchNorm) and m not in exclude:
            m._params = ParameterDict()
            m.hybrid_forward = types.MethodType(_forward, m)
    net.apply(_bypass)


def _merge_bn(net, conv_name="conv", bn_name="batchnorm", exclude=[]):
    def _collect_conv(m):
        if not hasattr(_collect_conv, "convs"):
            _collect_conv.convs = []
        if isinstance(m, nn.Conv2D):
            _collect_conv.convs.append(m)
    net.apply(_collect_conv)
    conv_lst = _collect_conv.convs

    bn_names = [c.name.replace(conv_name, bn_name) for c in conv_lst]
    for conv, bn in zip(conv_lst, bn_names):
        params = net.collect_params(bn)
        if len(params.keys()) != 0 and conv not in exclude:
            gamma = params[bn + "_gamma"].data()
            beta = params[bn + "_beta"].data()
            mean = params[bn + "_running_mean"].data()
            var = params[bn + "_running_var"].data()

            weight = conv.weight.data()
            w_shape = conv.weight.shape
            cout = w_shape[0]
            conv.weight.set_data( (weight.reshape(cout, -1) * gamma.reshape(-1, 1) \
                                  / nd.sqrt(var + 1e-5).reshape(-1, 1)).reshape(w_shape) )
            if conv.bias is None:
                conv._kwargs['no_bias'] = False
                conv.bias = conv.params.get('bias',
                                            shape=(cout,), init="zeros",
                                            allow_deferred_init=True)
                conv.bias.initialize()
            bias = conv.bias.data()
            conv.bias.set_data(gamma * (bias - mean) / nd.sqrt(var + 1e-5) + beta)


def merge_bn(net, conv_name="conv", bn_name="batchnorm", exclude=[]):
    _merge_bn(net, conv_name, bn_name, exclude)
    _bypass_bn(net, exclude)
