#-*- coding: utf-8 -*-

import sys
sys.path.append("..")

from mxnet import nd
from mxnet.gluon.model_zoo.vision import vgg16
from quantize.convert import convert_model
from quantize.initialize import qparams_init


def test_qparams_init():
    net = vgg16(pretrained=True)
    convert_model(net)
    qparams_init(net)
    input = nd.random.uniform(shape=(1,3,224,224))
    print(net(input))


if __name__ == "__main__":
    test_qparams_init()