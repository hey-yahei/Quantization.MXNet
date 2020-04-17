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

from mxnet import nd

__all__ = ['Winograd_G']


_G23 = [
    [1, 0, 0],
    [1/2, 1/2, 1/2],
    [1/2, -1/2, 1/2],
    [0, 0, 1]
]

_G43 = [
    [1/4, 0, 0],
    [-1/6, -1/6, -1/6],
    [-1/6, 1/6, -1/6],
    [1/24, 1/12, 1/6],
    [1/24, -1/12, 1/6],
    [0, 0, 1]
]

_G63 = [
    [1, 0, 0],
    [-2/9, -2/9, -2/9],
    [-2/9, 2/9, -2/9],
    [1/90, 1/45, 2/45],
    [1/90, -1/45, 2/45],
    [32/45, 16/45, 8/45],
    [32/45, -16/45, 8/45],
    [0, 0, 1]
]

Winograd_G = {
    "F23": nd.array(_G23),
    "F43": nd.array(_G43),
    "F63": nd.array(_G63)
}
