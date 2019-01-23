#-*- coding: utf-8 -*-

from mxnet import nd
from mxnet.initializer import Constant

__all__ = ["qparams_init"]


def qparams_init(net, conv_name="conv", bn_name="batchnorm"):
    """
    Initialize quantized parameters for convolution op
    :param net: mxnet.gluon.nn.Block
        The net to initialize.
    :return: mxnet.gluon.nn.Block
        The net that has been initialized.
    """
    quant_params = net.collect_params()
    conv_params = [pname[:-len("_weight_min")] for pname in quant_params if pname.endswith("_weight_min")]
    input_params = [pname[:-len("_input_min")] for pname in quant_params if pname.endswith("_input_min")]
    fakeconv_params = [pname[:-len("_gamma")] for pname in quant_params if pname.endswith("_gamma") and conv_name in pname]
    for conv in conv_params:
        weights = quant_params[conv + "_weight"].data()
        # If fake bn, recalculate weights and initialize some related params
        if conv in fakeconv_params:
            # Get params of batchnorm
            gamma = quant_params[conv.replace(conv_name, bn_name) + "_gamma"].data()
            beta = quant_params[conv.replace(conv_name, bn_name) + "_beta"].data()
            mean = quant_params[conv.replace(conv_name, bn_name) + "_running_mean"].data()
            var = quant_params[conv.replace(conv_name, bn_name) + "_running_var"].data()

            # Merge bn to conv
            w_shape = weights.shape
            cout = w_shape[0]
            weights = (weights.reshape(cout, -1) * gamma.reshape(-1, 1) / nd.sqrt(var + 1e-5).reshape(-1,1)).reshape(w_shape)
            bias_name = conv + "_bias"
            if bias_name in quant_params:
                try:
                    quant_params[bias_name].set_data(gamma * (quant_params[bias_name] - mean) / nd.sqrt(var + 1e-5) + beta)
                except AssertionError:
                    quant_params[bias_name].initialize(Constant(gamma * (-mean) / nd.sqrt(var + 1e-5) + beta))

            # Store params of bn at conv
            quant_params[conv + "_gamma"].initialize(Constant(gamma))
            quant_params[conv + "_beta"].initialize(Constant(beta))
            quant_params[conv + "_running_mean"].initialize(Constant(mean))
            quant_params[conv + "_running_var"].initialize(Constant(var))
        
        max = nd.max(weights).asscalar()
        min = nd.min(weights).asscalar()
        quant_params[conv + "_weight_min"].initialize( Constant(min) )
        quant_params[conv + "_weight_max"].initialize( Constant(max) )
    for ipt in input_params:
        quant_params[ipt + "_input_min"].initialize( Constant(0) )
        quant_params[ipt + "_input_max"].initialize( Constant(0) )

    return net


