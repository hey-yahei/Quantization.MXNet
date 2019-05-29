#-*- coding: utf-8 -*-

import types

from mxnet.gluon import nn
from .convert_conv2d import gen_conv2d_converter
from .convert_act import convert_relu_to_relu6
from .convert_bn import bypass_bn

__all__ = ["convert_model", "convert_to_relu6", 'default_convert_fn']
__author__ = "YaHei"

default_convert_fn = {
    nn.Conv2D: gen_conv2d_converter(),
    nn.Activation: None, # convert_relu_to_relu6,
    nn.BatchNorm: None # bypass_bn
}


def convert_model(net, exclude=[], convert_fn=default_convert_fn):
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
            m_type = type(m)
            fn = convert_fn.get(m_type)
            if fn is not None:
                fn(m)
    net.apply(_convert)

    # Add method to update ema for `input_min` and `input_max` in convs
    def _update_ema(self, momentum=0.99):
        for qconv in self.collect_quantized_convs():
            if getattr(qconv, "input_max", None) is not None:
                qconv.input_max.set_data((1 - momentum) * qconv.current_input_max + momentum * qconv.input_max.data())
            if getattr(qconv, "running_mean", None) is not None:
                qconv.running_mean.set_data((1 - momentum) * qconv.current_mean + momentum * qconv.running_mean.data())
            if getattr(qconv, "running_var", None) is not None:
                qconv.running_var.set_data((1 - momentum) * qconv.current_var + momentum * qconv.running_var.data())
    net.update_ema = types.MethodType(_update_ema, net)

    # Add a method to collect all quantized convolution blocks
    def _collect_quantized_convs(self):
        convs = []
        def _collect_convs(m):
            if isinstance(m, nn.Conv2D) and hasattr(m, 'quantize_args'):
                convs.append(m)
        net.apply(_collect_convs)
        return convs
    net.collect_quantized_convs = types.MethodType(_collect_quantized_convs, net)

    # Add method to control the mode of input quantization as online or offline
    def _quantize_input_offline(self):
        for qconv in self.collect_quantized_convs():
            qconv.quantize_input_offline = True
    def _quantize_input_online(self):
        for qconv in self.collect_quantized_convs():
            qconv.quantize_input_offline = False
    net.quantize_input_offline = types.MethodType(_quantize_input_offline, net)
    net.quantize_input_online = types.MethodType(_quantize_input_online, net)


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
