#-*- coding: utf-8 -*-

import types
from collections import namedtuple
from mxnet.gluon.nn import Activation

__all__ = ["convert_relu_to_relu6", 'gen_act_converter']
__author__ = "YaHei"


QuantizedArgs = namedtuple("ActQuantizedArgs", "width quantize_act")


def _relu6_forward(self, F, x):
    return F.clip(F.Activation(x, act_type=self._act_type, name='fwd'), 0., 6.)


def convert_relu_to_relu6(m):
    assert isinstance(m, Activation) and m._act_type == "relu"
    m.hybrid_forward = types.MethodType(_relu6_forward, m)


def _act_forward(self, F, x, act_max=None):
    # Normal Activation
    act = self.origin_forward(F, x)

    # Simulate quantization
    if self.quantize_args.quantize_act:
        self.current_act_max = F.max(act).asscalar()
        max_ = act_max.asscalar() if self.quantize_act_offline else self.current_act_max
        scale = max_ / (2 ** self.quantize_args.width - 1)
        act = (act.clip(0., max_) / scale).round() * scale

    return act


def _add_quantize_act_params(m):
    m.quantize_act_offline = False
    m.current_act_max = 0.
    m.act_max = m.params.get("act_max",
                             shape=(1,), init="zeros",
                             allow_deferred_init=True,
                             differentiable=False)


def gen_act_converter(width=8, quantize_act=True):
    def _converter(m):
        assert isinstance(m, Activation)

        _add_quantize_act_params(m)

        m.origin_forward = m.hybrid_forward
        m.hybrid_forward = types.MethodType(_act_forward, m)
        m.quantize_args = QuantizedArgs(width=width, quantize_act=quantize_act)
    return _converter


