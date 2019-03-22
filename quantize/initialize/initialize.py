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
    conv_params = net.collect_params(".*conv.*")
    bn_params = net.collect_params(".*batchnorm.*")
    conv_prefix = [pname[:-len("_weight")] for pname in conv_params if pname.endswith("_weight")]
    weight_params = [pname[:-len("_weight_min")] for pname in conv_params if pname.endswith("_weight_min")]
    input_params = [pname[:-len("_input_min")] for pname in conv_params if pname.endswith("_input_min")]
    fakeconv_params = [pname[:-len("_gamma")] for pname in conv_params if pname.endswith("_gamma") and conv_name in pname]
    for conv in conv_prefix:
        weights = conv_params[conv + "_weight"].data()
        # If fake bn, recalculate weights and initialize some related params
        if conv in fakeconv_params:
            # Get params of batchnorm
            gamma = bn_params[conv.replace(conv_name, bn_name) + "_gamma"].data()
            beta = bn_params[conv.replace(conv_name, bn_name) + "_beta"].data()
            mean = bn_params[conv.replace(conv_name, bn_name) + "_running_mean"].data()
            var = bn_params[conv.replace(conv_name, bn_name) + "_running_var"].data()

            # Merge bn to conv
            w_shape = weights.shape
            cout = w_shape[0]
            weights = (weights.reshape(cout, -1) * gamma.reshape(-1, 1) / nd.sqrt(var + 1e-5).reshape(-1,1)).reshape(w_shape)
            bias_name = conv + "_bias"
            if bias_name in conv_params:
                try:
                    conv_params[bias_name].set_data(gamma * (conv_params[bias_name] - mean) / nd.sqrt(var + 1e-5) + beta)
                except AssertionError:
                    conv_params[bias_name].initialize(Constant(gamma * (-mean) / nd.sqrt(var + 1e-5) + beta))

            # Store params of bn at conv
            conv_params[conv + "_gamma"].initialize(Constant(gamma))
            conv_params[conv + "_beta"].initialize(Constant(beta))
            conv_params[conv + "_running_mean"].initialize(Constant(mean))
            conv_params[conv + "_running_var"].initialize(Constant(var))

        if conv in weight_params:
            max = nd.max(weights).asscalar()
            min = nd.min(weights).asscalar()
            conv_params[conv + "_weight_min"].initialize( Constant(min) )
            conv_params[conv + "_weight_max"].initialize( Constant(max) )

        if conv in input_params:
            conv_params[conv + "_input_min"].initialize( Constant(0) )
            conv_params[conv + "_input_max"].initialize( Constant(0) )

