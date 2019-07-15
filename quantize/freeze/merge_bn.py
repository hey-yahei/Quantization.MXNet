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

from mxnet import nd
from mxnet.gluon.parameter import ParameterDict
from mxnet.gluon import nn

__all__ = ['merge_bn']
__author__ = "YaHei"


def _bypass_bn(net, exclude=[]):
    def _forward(self, F, x, *args, **kwargs):
        return x
    def _bypass(m):
        if isinstance(m ,nn.BatchNorm) and m not in exclude:
            # m._params = ParameterDict()
            m.hybrid_forward = types.MethodType(_forward, m)
    net.apply(_bypass)


def _merge_bn(net, conv_name="conv", bn_name="batchnorm", exclude=[]):
    conv_lst = []
    def _collect_conv(m):
        if isinstance(m, nn.Conv2D):
            assert not hasattr(m, "gamma"), "Don't merge bn to a conv with fake bn! ({})".format(m.name)
            conv_lst.append(m)
    net.apply(_collect_conv)

    bn_names = [c.name.replace(conv_name, bn_name) for c in conv_lst]
    for conv, bn in zip(conv_lst, bn_names):
        params = net.collect_params(bn + "_")
        if len(params.keys()) != 0 and conv not in exclude:
            print("Merge {} to {}".format(bn, conv.name))
            gamma = params[bn + "_gamma"].data()
            beta = params[bn + "_beta"].data()
            mean = params[bn + "_running_mean"].data()
            var = params[bn + "_running_var"].data()

            weight = conv.weight.data()
            w_shape = conv.weight.shape
            cout = w_shape[0]
            conv.weight.set_data( (weight.reshape(cout, -1) * gamma.reshape(-1, 1) \
                                  / nd.sqrt(var + 1e-10).reshape(-1, 1)).reshape(w_shape) )
            if conv.bias is None:
                conv._kwargs['no_bias'] = False
                conv.bias = conv.params.get('bias',
                                            shape=(cout,), init="zeros",
                                            allow_deferred_init=True)
                conv.bias.initialize()
            bias = conv.bias.data()
            conv.bias.set_data(gamma * (bias - mean) / nd.sqrt(var + 1e-10) + beta)


def merge_bn(net, conv_name="conv", bn_name="batchnorm", exclude=[]):
    """
    Merge all batchnorm to convolution.
    :param net: mxnet.gluon.nn.Block
        The net to merge bn for.
    :param conv_name: str
        The keyword in name of convolutions.
    :param bn_name: str
        The keyword in name of batchnorms.
    :param exclude: list of mxnet.gluon.nn.Conv2D
        The convolutions to exclude.
    :return: mxnet.gluon.nn.Block
        The net that has been merged bn.
    """
    _merge_bn(net, conv_name, bn_name, exclude)
    _bypass_bn(net, exclude)
