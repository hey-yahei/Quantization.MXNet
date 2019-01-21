#-*- coding: utf-8 -*-

import sys
sys.path.append("..")

from mxnet import nd
from mxnet.gluon.nn import Conv2D
from quantize.convert import convert_conv2d


if __name__ == "__main__":
    inputs = nd.zeros(shape=(2,3,10,20))
    # conv = Conv2D(channels=10, in_channels=3, strides=1, padding=1, use_bias=True, kernel_size=5)
    conv = Conv2D(channels=10, in_channels=3, strides=1, padding=1, use_bias=True, kernel_size=5, activation="relu")

    print("New instance")
    try:
        print("shape:", conv(inputs).shape)
    except Exception as e:
        print(e)

    print()
    print("Initialize origin module")
    conv.initialize()
    try:
        print("shape:", conv(inputs).shape)
    except Exception as e:
        print(e)

    print()
    print("Convert to quantized module")
    convert_conv2d(conv)
    try:
        print("shape:", conv(inputs).shape)
    except Exception as e:
        print(e)

    print()
    print("Initialize quantized parameters for converted module")
    params = conv.collect_params()
    for pname in params:
        if pname.endswith(("min", "max")):
            params[pname].initialize()
    try:
        print("shape:", conv(inputs).shape)
        print("w_min:", conv.w_min)
        print("w_max:", conv.w_max)
    except Exception as e:
        print(e)
