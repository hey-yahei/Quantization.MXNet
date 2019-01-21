#-*- coding: utf-8 -*-

from collections import OrderedDict

__all__ = ['collect_qparams', 'print_all_qparams']


def collect_qparams(net):
    """
    Collect all parameters about quantization.
    :param net: mxnet.gluon.nn.Block
        The net whose parameters would be scanned.
    :return: collections.OrderedDict with (pname, param) as key-value pairs
        `pname`: str
            The name of parameters.
        `param`: mxnet.gluon.parameter.Parameter
            The parameter whose name is `pname`.
    """
    ret = OrderedDict()
    quant_params = net.collect_params(".*[min|max]")
    for param in quant_params:
        if param.endswith(("_min", "_max")):
            ret[param] = quant_params[param]
    return ret


def print_all_qparams(net):
    qparams = collect_qparams(net)
    for param in qparams:
        print("{}:\t\t{:+.4f}".format(param, qparams[param].data().asscalar()))


