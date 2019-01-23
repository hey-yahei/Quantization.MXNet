#-*- coding: utf-8 -*-

import types

from mxnet.gluon import nn

__all__ = ['bypass_bn']
__author__ = "YaHei"


def bypass_bn(m):
    assert isinstance(m, nn.BatchNorm)
    def _forward(self, F, x, *args, **kwargs):
        return x
    m.hybrid_forward = types.MethodType(_forward, m)
