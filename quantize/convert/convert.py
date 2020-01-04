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

from mxnet.gluon import nn
from .convert_conv2d import gen_conv2d_converter
from .convert_dense import gen_dense_converter
from .convert_act import gen_act_converter, convert_relu_to_relu6
from .convert_bn import bypass_bn

__all__ = ["convert_model", "convert_to_relu6", 'default_convert_fn']
__author__ = "YaHei"

default_convert_fn = {
    nn.Conv2D: gen_conv2d_converter(),
    nn.Dense: gen_dense_converter(),
    nn.Activation: None, # convert_relu_to_relu6,  # gen_act_converter(),
    nn.BatchNorm: None # bypass_bn
}


def convert_model(net, exclude=[], convert_fn=default_convert_fn, custom_fn={}):
    """
    Convert the model to the one with simulated quantization.
    :param net: mxnet.gluon.nn.Block
        The net to convert.
    :param exclude: list of mxnet.gluon.nn.Block
        Blocks that want to exclude.
    :param convert_fn: dict with (module, func) key-value pairs
        `module`: mxnet.gluon.nn.Block
        `func`: function `func(module) -> None`
        Apply the function(value) to corresponding blocks(key).
    :return: mxnet.gluon.nn.Block
        The net that has been converted.
    """
    # Convert network
    def _convert(m):
        if m not in exclude:
            fn = custom_fn[m] if m in custom_fn else convert_fn.get(type(m))
            if fn is not None:
                fn(m)
    net.apply(_convert)

    # Add method to update ema for `input_min` and `input_max` in convs
    def _update_ema(self, momentum=0.9):
        for qblocks in self.collect_quantized_blocks():
            # if quantize input
            if getattr(qblocks, "input_max", None) is not None:
                qblocks.input_max.set_data((1 - momentum) * qblocks.current_input_max + momentum * qblocks.input_max.data())
            # if quantize activation
            if getattr(qblocks, "act_max", None) is not None:
                qblocks.act_max.set_data((1 - momentum) * qblocks.current_act_max + momentum * qblocks.act_max.data())
            # if fake bn
            if getattr(qblocks, "running_mean", None) is not None:
                qblocks.running_mean.set_data((1 - momentum) * qblocks.current_mean + momentum * qblocks.running_mean.data())
            if getattr(qblocks, "running_var", None) is not None:
                qblocks.running_var.set_data((1 - momentum) * qblocks.current_var + momentum * qblocks.running_var.data())
    net.update_ema = types.MethodType(_update_ema, net)

    # Add a method to collect all quantized convolution blocks
    def _collect_quantized_blocks(self):
        blocks = []
        def _collect_blocks(m):
            if type(m) in (nn.Dense, nn.Conv2D, nn.Activation) and hasattr(m, 'quantize_args'):
                blocks.append(m)
        net.apply(_collect_blocks)
        return blocks
    net.collect_quantized_blocks = types.MethodType(_collect_quantized_blocks, net)

    # Add method to control the mode of input quantization as online or offline
    def _quantize_input(self, enable=True, online=True):
        for qblocks in self.collect_quantized_blocks():
            if type(qblocks) in (nn.Dense, nn.Conv2D):
                assert (not enable) or qblocks.quantize_args.quantize_input
                qblocks.quantize_input = enable
                qblocks.quantize_input_offline = not online
            elif type(qblocks) == nn.Activation:
                assert (not enable) or qblocks.quantize_args.quantize_act
                qblocks.quantize_act = enable
                qblocks.quantize_act_offline = not online
    net.quantize_input = types.MethodType(_quantize_input, net)

    # Add method to control enable/disable quantization
    def _enable_quantize(self):
        for qblocks in self.collect_quantized_blocks():
            if type(qblocks) in (nn.Dense, nn.Conv2D, nn.Activation):
                qblocks.enable_quantize = True
    def _disable_quantize(self):
        for qblocks in self.collect_quantized_blocks():
            if type(qblocks) in (nn.Dense, nn.Conv2D, nn.Activation):
                qblocks.enable_quantize = False
    net.enable_quantize = types.MethodType(_enable_quantize, net)
    net.disable_quantize = types.MethodType(_disable_quantize, net)


def convert_to_relu6(net, exclude=[]):
    """
    Convert ReLUs in net to ReLU6.
    :param net: mxnet.gluon.nn.Block
        The net to convert.
    :param exclude: list of mxnet.gluon.nn.Activation
        ReLUs blocks that want to exclude.
    :return: mxnet.gluon.nn.Block
        The net that has been converted.
    """
    def _convert_to_relu6(m):
        if isinstance(m, nn.Activation) and m._act_type == "relu" and m not in exclude:
            convert_relu_to_relu6(m)
    return net.apply(_convert_to_relu6)
