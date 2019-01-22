#-*- coding: utf-8 -*-

import sys
sys.path.append("..")

import time

from mxnet import nd
from mxnet.gluon.model_zoo.vision import mobilenet1_0

from quantize.convert import convert_model
from quantize.initialize import qparams_init as qinit
from quantize.freeze import merge_bn
from quantize.freeze.merge_bn import _merge_bn


def test_check_output():
    print("<<TEST: Check whether output is right>>")
    inputs = nd.uniform(shape=(1,3,224,224))

    net = mobilenet1_0(pretrained=True)
    conv = net.features[0]
    bn = net.features[1]
    origin_output = bn( conv(inputs) )

    merge_bn(net)
    new_output = conv(inputs)

    cout = origin_output.shape[1]
    diff = new_output - origin_output
    print("Per channel ----")
    print("Diff max:", diff.reshape(cout, -1).max(axis=1))
    print("Diff mean:", diff.reshape(cout, -1).mean(axis=1))
    print("All ----")
    print("Diff max:", diff.max())
    print("Diff mean:", diff.mean())
    print()


def test_export():
    print("<<TEST: hybridize and export>>")
    bn_string_in_json_file = '"op": "BatchNorm"'
    export_file_name = "/tmp/test_merge_bn-" + time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))

    net = mobilenet1_0(pretrained=True)
    merge_bn(net)
    print("merge_bn ...[ok]")
    net.hybridize()
    print("hybridize ...[ok]")
    _ = net(nd.zeros(shape=(1, 3, 224, 224)))
    print("run hybrid graph forward ...[ok]")
    net.export(export_file_name, 0)
    print("export to", export_file_name, "...[ok]")
    with open(export_file_name+"-symbol.json", "r") as f:
        s = f.read()
        if bn_string_in_json_file not in s:
            print('[OK] op "BatchNorm" is not in exported file')
        else:
            print('[Error] op "BatchNorm" is in exported file')
    print()


def test_merge_bn():
    net = mobilenet1_0(pretrained=True)
    convert_model(net)
    qinit(net)
    _merge_bn(net)


if __name__ == "__main__":
    # test_check_output()
    # test_export()
    test_merge_bn()
