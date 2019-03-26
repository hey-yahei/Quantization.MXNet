#-*- coding: utf-8 -*-

from mxnet import nd
from mxnet.gluon import nn
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
    convs = []
    def _collect_convs(m):
        if isinstance(m, nn.Conv2D):
            convs.append(m)
    net.apply(_collect_convs)

    params = net.collect_params()
    for m in convs:
        conv_name = m.name
        weight = m.weight.data()

        # If fake bn, recalculate weight and initialize some related params
        if hasattr(m, "gamma"):
            # Get params of batchnorm
            gamma = params[conv_name.replace(conv_name, bn_name) + "_gamma"].data()
            beta = params[conv_name.replace(conv_name, bn_name) + "_beta"].data()
            mean = params[conv_name.replace(conv_name, bn_name) + "_running_mean"].data()
            var = params[conv_name.replace(conv_name, bn_name) + "_running_var"].data()

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
            weight = (weight.reshape(cout, -1) * gamma.reshape(-1, 1) / nd.sqrt(var + 1e-5).reshape(-1, 1)).reshape(w_shape)
            # bias = gamma * (m.bias.data() - mean) / nd.sqrt(var + 1e-5) + beta

        # Initialize for weight_min and weight_max
        if m.quantize_kwargs['weight']['training']:
            if m.quantize_kwargs['weight']['signed']:
                weight_flt = weight.reshape(-1)
                max = weight_flt.abs().sort()[int(weight_flt.shape[0] * 0.99)]
                # max = weight.abs().max()
                min = -max
            elif m.quantize_kwargs['weight']['one_side']:
                max = weight.max()
                min = None
            else:
                max = weight.max()
                min = weight.min()
            m.weight_max.initialize(Constant(max))
            print(m.name, "weight_max", max.asscalar())
            if min is not None:
                m.weight_min.initialize(Constant(min))

        if getattr(m, "input_min", None) is not None:
            m.input_min.initialize(Constant(0))
        if getattr(m, "input_max", None) is not None:
            m.input_max.initialize(Constant(0))


