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

from mxnet import nd
from mxnet.gluon import nn
from mxnet.initializer import Constant

__all__ = ["qparams_init"]


def qparams_init(net, conv_name="conv", bn_name="batchnorm"):
    """
    Initialize quantized parameters for convolution op
    :param net: mxnet.gluon.nn.Block
        The net to initialize.
    :param conv_name: str
    :param bn_name: str
    :return: mxnet.gluon.nn.Block
        The net that has been initialized.
    """
    blocks = net.collect_quantized_blocks()
    params = net.collect_params()

    for m in blocks:
        # If fake bn, recalculate weight and initialize some related params
        if isinstance(m, nn.Conv2D) and hasattr(m, "gamma"):
            name = m.name
            weight = m.weight.data()

            # Get params of batchnorm
            gamma = params[name.replace(conv_name, bn_name) + "_gamma"].data()
            beta = params[name.replace(conv_name, bn_name) + "_beta"].data()
            mean = params[name.replace(conv_name, bn_name) + "_running_mean"].data()
            var = params[name.replace(conv_name, bn_name) + "_running_var"].data()

            # Store params of bn at conv
            m.gamma.initialize(Constant(gamma))
            m.beta.initialize(Constant(beta))
            m.running_mean.initialize(Constant(mean))
            m.running_var.initialize(Constant(var))

            # Enable bias if need, and recalculate weight and bias with fake bn version
            w_shape = weight.shape
            cout = w_shape[0]
            if m.bias is None:
                m._kwargs['no_bias'] = False
                m.bias = m.params.get('bias',
                                      shape=(cout,), init="zeros",
                                      allow_deferred_init=True)
                m.bias.initialize()

        if type(m) in (nn.Conv2D, nn.Dense) and m.quantize_args.quantize_input:
            m.input_max.initialize(Constant(0))
        if type(m) == nn.Activation and m.quantize_args.quantize_act:
            m.act_max.initialize(Constant(0))

