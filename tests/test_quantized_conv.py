#-*- coding: utf-8 -*-

from mxnet import nd

import sys
sys.path.append("..")
from nn.quantized_conv import _im2col_2D, Conv2D as MyConv
from mxnet.gluon.nn import Conv2D
from mxnet import init
from quantize.convert import gen_conv2d_converter, convert_model
from quantize.initialize import qparams_init


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
    my_res = my_conv(x)

    ref_conv = Conv2D(10, 3, 1, 1, in_channels=2, groups=groups, use_bias=use_bias,
                      bias_initializer=init.Constant(my_conv.bias.data()) if use_bias else 'zero',
                      weight_initializer=init.Constant(my_conv.weight.data()))
    ref_conv.initialize()
    ref_res = ref_conv(x)

    sim_conv = ref_conv
    convert_conv2d = gen_conv2d_converter()
    convert_conv2d(sim_conv)
    qparams_init((sim_conv))
    sim_res = sim_conv(x)

    return (((my_res - ref_res).abs() / ref_res.abs()).reshape(-1).sort()[-20:],
            ((my_res - sim_res).abs() / sim_res.abs()).reshape(-1).sort()[-20:])


def test_quantized_mobilnet():
    x = nd.uniform(shape=(1,3,224,224))

    from models.quantized_mobilenet import mobilenet1_0 as my_mobilenet
    my_net = my_mobilenet(pretrained=True)
    my_res = my_net(x)

    from gluoncv.model_zoo import mobilenet1_0 as ref_mobilenet
    ref_net = ref_mobilenet(pretrained=True)
    ref_res = ref_net(x)

    sim_net = ref_mobilenet(pretrained=True)
    convert_model(sim_net, exclude=[sim_net.features[0]])
    qparams_init(sim_net)
    sim_res = sim_net(x)

    # print(">> Difference between my_quantized_mobilenet and simulated_mobilenet -- ")
    # print(">> ", ((my_res - sim_res).abs() / sim_res.abs()).reshape(-1).sort()[-100:])

    return (my_res.argsort(is_ascend=False)[0, :20],
            ref_res.argsort(is_ascend=False)[0, :20],
            sim_res.argsort(is_ascend=False)[0, :20])


if __name__ == "__main__":
    # test_im2col_2D()

    # print("Difference between my_conv and ref_conv -- ")
    # print("no bias, group=1: ", test_Conv2D(use_bias=False, groups=1))
    # print("bias, group=1   : ", test_Conv2D(use_bias=True, groups=1))
    # print("no bias, group=2: ", test_Conv2D(use_bias=False, groups=2))
    # print("bias, group=2   : ", test_Conv2D(use_bias=True, groups=2))

    print("Difference between my_quantized_conv and ref_conv -- (error ratio[Top20]) ")
    print("no bias, group=1: ", test_quantized_Conv2D(use_bias=False, groups=1))
    print("bias, group=1   : ", test_quantized_Conv2D(use_bias=True, groups=1))
    print("no bias, group=2: ", test_quantized_Conv2D(use_bias=False, groups=2))
    print("bias, group=2   : ", test_quantized_Conv2D(use_bias=True, groups=2))

    # print("Difference between my_quantized_mobilenet and ref_mobilenet -- ")
    # my_pred, ref_pred, sim_pred = test_quantized_mobilnet()
    # print("My prediction [Top20]: ", my_pred)
    # print("Reference prediction [Top20]: ", ref_pred)
    # print("Simulation prediction [Top20]: ", sim_pred)
