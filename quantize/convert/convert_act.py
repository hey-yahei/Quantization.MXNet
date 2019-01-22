#-*- coding: utf-8 -*-

import types
from mxnet.gluon.nn import Activation

__all__ = ["convert_relu_to_relu6"]
__author__ = "YaHei"


def _relu6_forward(self, F, x):
    return F.clip(F.Activation(x, act_type=self._act_type, name='fwd'), 0., 6.)


def convert_relu_to_relu6(m):
    assert isinstance(m, Activation) and m._act_type == "relu"
    m.hybrid_forward = types.MethodType(_relu6_forward, m)
