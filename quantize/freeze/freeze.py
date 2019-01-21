#-*- coding: utf-8 -*-

import ctypes
import json
import os
import functools
from mxnet import symbol, nd, model
from mxnet.base import _LIB, check_call
from mxnet.base import c_array, c_str, mx_uint
from mxnet.base import SymbolHandle
from mxnet.symbol import Symbol

__all__ = ['quantize_symbol', 'quantize_params', 'FreezeHelper']


def _quantize_symbol(sym, excluded_symbols=[], offline_params=[],
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
                     quantized_dtype='uint8', calib_quantize_op=False, quantize_input_offline=True):
    """
    Quantize symbol.
    :param sym: mxnet.symbol.Symbol
        The symbol to quantize.
    :param excluded_symbols: list of mxnet.base.SymbolHandle
        The handle of symbols to exclude.
    :param offline_params: list of str
        The names of parameters to quantize offline.
    :param quantized_dtype: str {"int8", "uint8"}
        The data type that you will quantize to.
    :param calib_quantize_op: bool
        Calibrate or not.(Only for quantization online.
    :param quantize_input_offline: bool
        Quantize the input of blocks such as convs, pools or not.
    :return: mxnet.symbol.Symbol
        The symbol that has been quantized.
    """
    sym = _quantize_symbol(sym=sym,
                           excluded_symbols=excluded_symbols,
                           offline_params=offline_params,
                           quantized_dtype=quantized_dtype,
                           calib_quantize_op=calib_quantize_op)

    if quantize_input_offline:
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
                                                       out_type="uint8")
            quantized_params[name] = val
            quantized_params[name+'_min'] = vmin
            quantized_params[name+'_max'] = vmax
        elif name in params:
            quantized_params[name] = params[name]
    return quantized_params


class FreezeHelper(object):
    def __init__(self, net, params_filename, input_shape, tmp_filename="tmp_origin_net"):
        """
        A helper for freezing.
        :param net: mxnet.gluon.nn.Block
            The origin net that you want to load trained parameters and freeze.
        :param params_filename: str
            The filename of the trained parameters.
        :param input_shape: tuple
            The shape of input. For example, (1, 3, 224, 224) for MobileNet.
        :param tmp_filename: str
            The filename to output as temporary file.
        """
        self.origin_net = net
        self.gluon_params_filename = params_filename
        self.tmp_filename = tmp_filename
        self.sym, self.args, self.auxes = None, None, None

        net.load_parameters(params_filename, ignore_extra=True)
        net.hybridize()
        _ = net(nd.zeros(shape=input_shape))
        net.export(self.tmp_filename, 0)
        self.sym, self.args, self.auxes = model.load_checkpoint(self.tmp_filename, 0)

    def __del__(self):
        os.remove(self.tmp_filename + "-symbol.json")
        os.remove(self.tmp_filename + "-0000.params")

    def list_symbols(self, prefix="", suffix=""):
        return [s for s in self.sym.get_internals() if s.name.startswith(prefix) and s.name.endswith(suffix)]

    @staticmethod
    def _is_number(s):
        try:
            _ = int(s)
            return True
        except:
            return False

    def _extract_qparams(self, net, gluon_params_filename):
        gluon_params = nd.load(gluon_params_filename)
        quantized_params = {}
        for k in gluon_params.keys():
            *others, attr_name = k.split(".")
            atom_block = functools.reduce(
                lambda b, n: b[int(n)] if self._is_number(n) else getattr(b, n),
                others, net
            )
            if attr_name in ("weight_min", "weight_max", "input_min", "input_max"):
                quantized_params[atom_block.name + "_" + attr_name] = gluon_params[k]
        return quantized_params

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

        exclude = [s for s in self.sym.get_internals() if s.name in excluded_symbol]
        qsym = quantize_symbol(sym=self.sym,
                               excluded_symbols=exclude,
                               offline_params=offline_params,
                               quantized_dtype=quantized_dtype,
                               calib_quantize_op=calib_quantize_op,
                               quantize_input_offline=quantize_input_offline)
        quantized_params = self._extract_qparams(self.origin_net, self.gluon_params_filename)
        self.args.update(quantized_params)
        qargs = quantize_params(qsym, self.args)
        return qsym, qargs, self.auxes


