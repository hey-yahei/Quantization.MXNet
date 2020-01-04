#-*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2019 hey-yahei
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from mxnet import autograd

__all__ = ['LinearQuantizeSTE']
__author__ = 'YaHei'


class LinearQuantizeSTE(autograd.Function):
    def __init__(self, scale, clip_max=None, clip_min=None):
        super(LinearQuantizeSTE, self).__init__()
        self.clip_max = clip_max
        self.clip_min = clip_min if clip_min is not None else 0.
        self.scale = scale

    def forward(self, x):
        if self.clip_max is None:
            return (x / (self.scale + 1e-10)).round() * self.scale
        else:
            return (x.clip(self.clip_min, self.clip_max) / (self.scale + 1e-10)).round() * self.scale

    def backward(self, dy):
        return dy

