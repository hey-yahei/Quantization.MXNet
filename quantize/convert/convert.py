#-*- coding: utf-8 -*-

import types

from mxnet.gluon import nn
from .convert_conv2d import convert_conv2d_quantize_input
from .convert_act import convert_relu_to_relu6

__all__ = ["convert_model", "convert_to_relu6"]
__author__ = "YaHei"

default_convert_fn = {
    nn.Conv2D: convert_conv2d_quantize_input,
    nn.Activation: convert_relu_to_relu6
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
        if m in exclude:
            return
        m_type = type(m)
        fn = convert_fn.get(m_type)
        if fn is not None:
            fn(m)
            if m_type == nn.Conv2D:
                net.quantized_convs.append(m)
    net.quantized_convs = []
    net.apply(_convert)

    # Add method to update ema for `input_min` and `input_max` in convs
    def _update_ema(self, momentum=0.99):
        for qconv in self.quantized_convs:
            qconv.input_min.set_data( (1 - momentum) * qconv.current_input_min + momentum * qconv.input_min.data() )
            qconv.input_max.set_data( (1 - momentum) * qconv.current_input_max + momentum * qconv.input_max.data() )
    net.update_ema = types.MethodType(_update_ema, net)

    # Add method to control the mode of input quantization as online or offline
    def _quantize_input_offline(self):
        for qconv in self.quantized_convs:
            qconv.quantize_input_offline = True
    def _quantize_input_online(self):
        for qconv in self.quantized_convs:
            qconv.quantize_input_offline = False
    net.quantize_input_offline = types.MethodType(_quantize_input_offline, net)
    net.quantize_input_online = types.MethodType(_quantize_input_online, net)

    return net


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
