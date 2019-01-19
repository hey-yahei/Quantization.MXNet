#-*- coding: utf-8 -*-

from collections import OrderedDict

__all__ = ['collect_qparams', 'print_all_qparams']


def collect_qparams(net):
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


