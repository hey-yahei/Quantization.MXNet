#-*- coding: utf-8 -*-

import ctypes
import logging
import os
import numpy as np
import json
from mxnet import symbol
from mxnet.base import _LIB, check_call, py_str
from mxnet.base import c_array, c_str, mx_uint, c_str_array
from mxnet.base import NDArrayHandle, SymbolHandle
from mxnet.symbol import Symbol
from mxnet.symbol import load as sym_load
from mxnet import ndarray
from mxnet.ndarray import load as nd_load
from mxnet.ndarray import NDArray
from mxnet.io import DataIter
from mxnet.context import cpu, Context
from mxnet.module import Module


def quantize(data, min, max):
    scale = (max - min) / (2 ** 8 - 1)
    return ((data.clip(min, max) - min) / scale).round()


def _quantize_symbol(sym, excluded_symbols=[], offline_params=[],
                     quantized_dtype='uint8', calib_quantize_op=False):
    assert isinstance(excluded_symbols, list)
    num_excluded_symbols = len(excluded_symbols)
    exclude = [s.handle for s in excluded_symbols]

    assert isinstance(offline_params, list)
    offline = [c_str(k) for k in offline_params]
    num_offline = len(offline)

    out = SymbolHandle()
    check_call(_LIB.MXQuantizeSymbol(sym.handle,
                                     ctypes.byref(out),
                                     mx_uint(num_excluded_symbols),
                                     c_array(SymbolHandle, exclude),
                                     mx_uint(num_offline),
                                     c_array(ctypes.c_char_p, offline),
                                     c_str(quantized_dtype),
                                     ctypes.c_bool(calib_quantize_op)))
    return Symbol(out)


def quantize_symbol(sym, excluded_symbols=[], offline_params=[],
                     quantized_dtype='uint8', calib_quantize_op=False, quantize_input=True):
    sym = _quantize_symbol(sym=sym,
                           excluded_symbols=excluded_symbols,
                           offline_params=offline_params,
                           quantized_dtype=quantized_dtype,
                           calib_quantize_op=calib_quantize_op)

    if quantize_input:
        sym_json = json.loads(sym.tojson())
        for i, node in enumerate(sym_json['nodes']):
            if node['op'] == '_contrib_quantize':
                min_node_idx = i - 2
                max_node_idx = i - 1
            elif node['op'].startswith("_contrib_quantized_"):
                min_node = sym_json['nodes'][min_node_idx]
                max_node = sym_json['nodes'][max_node_idx]
                min_node.update({
                    "op": "null",
                    "name": node['name'][len("quantized_"):-len("_fwd")] + "_input_min",
                    "inputs": []
                })
                max_node.update({
                    "op": "null",
                    "name": node['name'][len("quantized_"):-len("_fwd")] + "_input_max",
                    "inputs": []
                })
        sym = symbol.load_json(json.dumps(sym_json))

    return sym


def quantize_params(qsym, params, th_dict={}):
    inputs_name = qsym.list_arguments()
    quantized_params = {}
    for name in inputs_name:
        if name.endswith('_quantize'):
            original_name = name[:-len('_quantize')]
            val, vmin, vmax = ndarray.contrib.quantize(data=params[original_name],
                                                       min_range=params[original_name+"_min"],
                                                       max_range=params[original_name+"_max"],
                                                       out_type="uint8")
            quantized_params[name] = val
            quantized_params[name+'_min'] = vmin
            quantized_params[name+'_max'] = vmax
        elif name in params:
            quantized_params[name] = params[name]
        # # ignore online quantize params?
        # elif name.endswith(('_min')):
        #     output = name[: - len('_min')]
        #     if output in th_dict:
        #         quantized_params[name] = ndarray.array([th_dict[output][0]])
        # elif name.endswith(('_max')):
        #     output = name[: - len('_max')]
        #     if output in th_dict:
        #         quantized_params[name] = ndarray.array([th_dict[output][1]])
    return quantized_params


