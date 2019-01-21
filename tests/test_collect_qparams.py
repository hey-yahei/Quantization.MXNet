#-*- coding: utf-8 -*-

import sys
sys.path.append("..")

from mxnet.gluon.model_zoo.vision import vgg16

from quantize import collect_qparams, print_all_qparams
from quantize.convert import convert_model
from quantize.initialize import qparams_init


def test_collect_qparams():
    net = vgg16(pretrained=True)
    convert_model(net)
    qparams_init(net)
    print("collect_qparams: ", collect_qparams(net))
    print("")
    print_all_qparams(net)


if __name__ == "__main__":
    test_collect_qparams()