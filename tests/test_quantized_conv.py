#-*- coding: utf-8 -*-

from mxnet import nd

import sys
sys.path.append("..")
from nn.quantized_conv import _im2col_2D, Conv2D as MyConv
from mxnet.gluon.nn import Conv2D
from mxnet import init

def test_im2col_2D():
    x = nd.uniform(shape=(2,3,5,5))
    print("Origin:")
    print(x)
    print("im2col_2D:")
    print(_im2col_2D(nd, x, (3,3), (1,1), (1,1)))

def test_Conv2D(use_bias, groups):
    x = nd.uniform(shape=(2,2,5,5))

    my_conv = MyConv(10, 3, 1, 1, in_channels=2, groups=groups, use_bias=use_bias)
    my_conv.initialize()

    ref_conv = Conv2D(10, 3, 1, 1, in_channels=2, groups=groups, use_bias=use_bias,
                      bias_initializer=init.Constant(my_conv.bias.data()) if use_bias else 'zero',
                      weight_initializer=init.Constant(my_conv.weight.data()))
    ref_conv.initialize()

    return (my_conv(x) - ref_conv(x)).abs().sum().asscalar()

def test_quantized_Conv2D(use_bias, groups):
    x = nd.uniform(shape=(2, 2, 5, 5))

    my_conv = MyConv(10, 3, 1, 1, in_channels=2, groups=groups, use_bias=use_bias,
                     input_dtype='uint8', weight_dtype='int8', quantized=True)
    my_conv.initialize()

    ref_conv = Conv2D(10, 3, 1, 1, in_channels=2, groups=groups, use_bias=use_bias,
                      bias_initializer=init.Constant(my_conv.bias.data()) if use_bias else 'zero',
                      weight_initializer=init.Constant(my_conv.weight.data()))
    ref_conv.initialize()

    my_res = my_conv(x)
    ref_res = ref_conv(x)
    return ((my_res - ref_res).abs() / ref_res.abs()).reshape(-1).sort()[-20:]

if __name__ == "__main__":
    # test_im2col_2D()

    # print("Difference between my_conv and ref_conv -- ")
    # print("no bias, group=1: ", test_Conv2D(use_bias=False, groups=1))
    # print("bias, group=1   : ", test_Conv2D(use_bias=True, groups=1))
    # print("no bias, group=2: ", test_Conv2D(use_bias=False, groups=2))
    # print("bias, group=2   : ", test_Conv2D(use_bias=True, groups=2))

    print("Difference between my_quantized_conv and ref_conv -- (error ratio[Top20]) ")
    print(test_quantized_Conv2D(use_bias=False, groups=1))
    print(test_quantized_Conv2D(use_bias=True, groups=1))
    print(test_quantized_Conv2D(use_bias=False, groups=2))
    print(test_quantized_Conv2D(use_bias=True, groups=2))
