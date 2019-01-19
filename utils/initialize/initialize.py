#-*- coding: utf-8 -*-

from mxnet import nd
from mxnet.initializer import Constant

__all__ = ["qparams_init"]


def qparams_init(net):
    # Initialize quantized parameters for convolution op
    quant_params = net.collect_params(".*[min|max]")
    conv_params = [pname[:-len("_weight_min")] for pname in quant_params if pname.endswith("_weight_min")]
    input_params = [pname[:-len("_input_min")] for pname in quant_params if pname.endswith("_input_min")]
    for conv in conv_params:
        weights = quant_params[conv + "_weight"].data()
        max = nd.max(weights).asscalar()
        min = nd.min(weights).asscalar()
        quant_params[conv + "_weight_min"].initialize( Constant(min) )
        quant_params[conv + "_weight_max"].initialize( Constant(max) )
    for ipt in input_params:
        quant_params[ipt + "_input_min"].initialize( Constant(0) )
        quant_params[ipt + "_input_max"].initialize( Constant(0) )
    return net


