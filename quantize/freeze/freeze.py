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

import ctypes
import json
import os
import functools
from collections import OrderedDict

import mxnet as mx
from mxnet import symbol, nd, model
from mxnet.base import _LIB, check_call
from mxnet.base import c_array, c_str, c_str_array, mx_uint
from mxnet.base import SymbolHandle
from mxnet.symbol import Symbol


__all__ = ['quantize_symbol', 'quantize_params', 'FreezeHelper']
__author__ = "YaHei"


def quantize_symbol(sym, excluded_symbols=[], offline_params=[],
                     quantized_dtype='uint8', calib_quantize_op=False):
    """
    Quantize symbol.
    :param sym: mxnet.symbol.Symbol
        The symbol to quantize.
    :param excluded_symbols: list of str
        The names of symbols to exclude.
    :param offline_params: list of str
        The names of parameters to quantize offline.
    :param quantized_dtype: {"int8", "uint8"}
        The data type that you will quantize to.
    :param calib_quantize_op: bool
        Calibrate or not.(Only for quantization online.
    :return: mxnet.symbol.Symbol
        The symbol that has been quantized.
    """
    assert isinstance(excluded_symbols, list)
    num_excluded_symbols = len(excluded_symbols)
    # exclude = [s.handle for s in excluded_symbols]

    assert isinstance(offline_params, list)
    offline = [c_str(k) for k in offline_params]
    num_offline = len(offline)

    out = SymbolHandle()
    check_call(_LIB.MXQuantizeSymbol(sym.handle,
                                     ctypes.byref(out),
                                     mx_uint(num_excluded_symbols),
                                     c_str_array(excluded_symbols),
                                     mx_uint(num_offline),
                                     c_array(ctypes.c_char_p, offline),
                                     c_str(quantized_dtype),
                                     ctypes.c_bool(calib_quantize_op)))
    return Symbol(out)


def quantize_params(qsym, params):
    """
    Quantize parameters.
    :param qsym: mxnet.symbol.Symbol
        The symbol that has been quantized by function `quantize_symbol`.
    :param params: dict with (pname, value) key-value pairs.
        `pname`: str
            The name of parameter.
        `value`: mxnet.nd.NDArry
            The value of parameter with dtype "float32".
    :return: dict with (pname, value) key-value pairs.
        `pname`: str
            The name of quantized parameter.
        `value`: mxnet.nd.NDArry
            The value of the parameter with dtype "uint8" or "int8".
    """
    inputs_name = qsym.list_arguments()
    quantized_params = {}
    for name in inputs_name:
        if name.endswith('_quantize'):
            original_name = name[:-len('_quantize')]
            val, vmin, vmax = nd.contrib.quantize(data=params[original_name],
                                                  min_range=params[original_name+"_min"],
                                                  max_range=params[original_name+"_max"],
                                                  out_type="int8")
            quantized_params[name] = val
            quantized_params[name+'_min'] = vmin
            quantized_params[name+'_max'] = vmax
        elif name in params:
            quantized_params[name] = params[name]
    return quantized_params


def calibrate_quantized_sym(qsym, th_dict):
    if th_dict is None or len(th_dict) == 0:
        return qsym
    num_layer_outputs = len(th_dict)
    layer_output_names = []
    min_vals = []
    max_vals = []
    for k, v in th_dict.items():
        layer_output_names.append(k)
        min_vals.append(v[0])
        max_vals.append(v[1])

    calibrated_sym = SymbolHandle()
    check_call(_LIB.MXSetCalibTableToQuantizedSymbol(qsym.handle,
                                                     mx_uint(num_layer_outputs),
                                                     c_str_array(layer_output_names),
                                                     c_array(ctypes.c_float, min_vals),
                                                     c_array(ctypes.c_float, max_vals),
                                                     ctypes.byref(calibrated_sym)))
    return Symbol(calibrated_sym)


class FreezeHelper(object):
    def __init__(self, net, params_filename):
        """
        A helper for freezing.
        :param net: mxnet.gluon.nn.Block
            The origin net that you want to load trained parameters and freeze.
        :param params_filename: str
            The filename of the trained parameters.
        # :param input_shape: tuple
        #     The shape of input. For example, (1, 3, 224, 224) for MobileNet.
        """
        self.origin_net = net
        self.gluon_params_filename = params_filename
        self.sym, self.args, self.auxes = None, None, None

        net.load_parameters(params_filename, ignore_extra=True)
        net.hybridize()
        x = mx.sym.var('data')
        y = net(x)
        y = mx.sym.SoftmaxOutput(data=y, name='softmax')
        self.sym = mx.symbol.load_json(y.tojson()).get_backend_symbol("MKLDNN")
        self.args = {}
        self.auxes = {}
        params = net.collect_params()
        # print(params)
        for param in params.values():
            v = param._reduce()
            k = param.name
            if 'running' in k:
                self.auxes[k] = v
            else:
                self.args[k] = v

    def list_symbols(self, prefix="", suffix=""):
        return [s for s in self.sym.get_internals() if s.name.startswith(prefix) and s.name.endswith(suffix)]

    @staticmethod
    def _is_number(s):
        try:
            _ = int(s)
            return True
        except:
            return False

    def _act_max_list(self):
        gluon_params = nd.load(self.gluon_params_filename)
        act_max_list = OrderedDict()
        for k in gluon_params.keys():
            *others, attr_name = k.split(".")
            if attr_name == "act_max":
                atom_block = functools.reduce(
                    lambda b, n: b[int(n)] if self._is_number(n) else getattr(b, n),
                    others, self.origin_net
                )
                act_max_list[f'{atom_block.name}'] = gluon_params[k].asscalar()
        return act_max_list

    def _set_min_max(self, sym):
        act_max = list( self._act_max_list().values() )
        sym_json = json.loads(sym.tojson())

        for i, node in enumerate(sym_json['nodes']):
            if node['op'] == '_contrib_quantize_v2':
                max_ = act_max.pop(0)
                print(f"{node['name']}: max_range: {max_}")
                node['attrs']['min_calib_range'] = 0.
                node['attrs']['max_calib_range'] = max_
            elif node['op'] == '_sg_mkldnn_conv' and node['attrs'].get('quantized', None) == 'true':
                max_ = act_max.pop(0)
                print(f"{node['name']}: max_range: {max_}")
                node['attrs']['min_calib_range'] = 0.
                node['attrs']['max_calib_range'] = max_
        assert act_max == []

        return mx.sym.load_json(json.dumps(sym_json))

    def freeze(self, excluded_symbol=[], offline_params=[], quantized_dtype="uint8",
               calib_quantize_op=False, quantize_input_offline=True):
        """
        Freeze the quantized model.
        :param excluded_symbol: list of str
            The names of symbols to exclude.
        :param offline_params: list of str
            Refer to `quantize_symbol` function.
        :param quantized_dtype: str {"int8", "uint8"}
            Refer to `quantize_symbol` function.
        :param calib_quantize_op: bool
            Refer to `quantize_symbol` function.
        :param quantize_input_offline: bool
            Refer to `quantize_symbol` function.
        :return:
        """
        qsym = quantize_symbol(sym=self.sym,
                               excluded_symbols=excluded_symbol,
                               offline_params=offline_params,
                               quantized_dtype=quantized_dtype,
                               calib_quantize_op=calib_quantize_op)
        qsym = qsym.get_backend_symbol("MKLDNN_QUANTIZE")
        qsym.save("./test.json")
        qsym = self._set_min_max(qsym)

        # quantized_params = self._extract_qparams(self.origin_net, self.gluon_params_filename)
        # self.args.update(quantized_params)
        qargs = quantize_params(qsym, self.args)
        return qsym, qargs, self.auxes


