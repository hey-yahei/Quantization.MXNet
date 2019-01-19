#-*- coding: utf-8 -*-

import sys
sys.path.append("..")

from mxnet import nd
from mxnet.gluon.model_zoo.vision import mobilenet1_0

from utils.convert import convert_conv2d_quantize_input
from utils.initialize import qparams_init as qinit
from utils.freeze import freeze_conv2d


if __name__ == "__main__":
    net = mobilenet1_0(pretrained=True)
    conv = net.features[0]
    convert_conv2d_quantize_input(conv)
    qinit(conv)
    freeze_conv2d(conv)
    conv.hybridize()
    conv(nd.zeros(shape=(1,3,224,224)))
    # print(conv.collect_params())
    conv.export("/tmp/freeze_conv2d_test", 0)

