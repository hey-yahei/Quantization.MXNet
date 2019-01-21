#-*- coding: utf-8 -*-

import sys
sys.path.append("..")

import time

from mxnet import sym, nd
from mxnet.gluon.model_zoo.vision import mobilenet1_0

from quantize.freeze import quantize_symbol


def test_quantize_symbol():
    print("<<TEST: Quantize symbol for mobilenet_v1_1_0>>")
    export_file_name = "/tmp/test_freeze_utils-" + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    in_file_name = export_file_name + "-symbol.json"
    out_file_name = export_file_name + "-qsymbol.json"

    net = mobilenet1_0(pretrained=True)
    net.hybridize()
    _ = net(nd.zeros(shape=(1, 3, 224, 224)))
    net.export(export_file_name, 0)

    mobilenet_sym = sym.load(in_file_name)
    qsym = quantize_symbol(mobilenet_sym)
    qsym.save()

    print('Quantized symbol has saved to ' + out_file_name)
    print()
    return out_file_name


def test_quantize_symbol_exclude():
    from mxnet import symbol
    sym = symbol.load("../examples/models/imagenet1k-inception-bn-symbol.json")
    conv1 = sym.get_internals()[3]
    qsym = quantize_symbol(sym, excluded_symbols=[conv1])
    print([s.name for s in qsym.get_internals()])


if __name__ == "__main__":
    # test_quantize_symbol()
    test_quantize_symbol_exclude()
