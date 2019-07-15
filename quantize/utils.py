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

from collections import OrderedDict

__all__ = ['collect_qparams', 'print_all_qparams']
__author__ = "YaHei"


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


