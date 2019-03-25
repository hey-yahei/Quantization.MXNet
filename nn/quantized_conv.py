#-*- coding: utf-8 -*-

from mxnet.gluon import nn

__all__ = ['Conv2D']
__author__ = "YaHei"

# Helper: convert int to tuple
def _int2tuple(x):
    return (x, ) * 2 if isinstance(x, int) else x

# Helper: im2col operation
def _im2col_2D(F, x, kernel_size, strides, padding):
    # Get parameters
    batch_size = x.shape[0]
    kh, kw = kernel_size
    sh, sw = strides
    ph, pw = padding
    # Pad
    x_pad = x.pad(mode='constant', constant_value=0, pad_width=(0,0,0,0,ph,ph,pw,pw))
    height, width = x_pad.shape[2], x_pad.shape[3]
    # Generate cols
    cols = []
    for h in range(0, height - kh + 1, sh):
        for w in range(0, width - kw + 1, sw):
            # Pick a window, reshape and append
            win = x_pad[:, :, h:(h + kh), w:(w + kw)]
            col = win.reshape((batch_size, -1))     # batch_size x (kh * kw * in_channels)
            cols.append(col)
    # Calculate size of output feature map
    out_h = (height - kh + 1) // sh
    out_w = (width - kw + 1) // sw
    # Stack all cols into a matrix
    return F.stack(*cols, axis=1), out_h, out_w    # batch_size x (out_height * out_width) x (kh * kw * in_channels)

class Conv2D(nn.HybridBlock):
    def __init__(self, channels, kernel_size, strides, padding, in_channels, groups=1,
                 activation=None, use_bias=True,
                 input_dtype='float32', weight_dtype='float32', bias_dtype='float32',
                 weight_initializer=None, bias_initializer='zero',
                 prefix=None, params=None):
        super(Conv2D, self).__init__(prefix, params)
        with self.name_scope():
            self._channels = channels
            self._in_channels = in_channels
            self._groups = groups
            assert in_channels % groups == 0 and channels % groups == 0
            self._kernel_size = _int2tuple(kernel_size)
            self._strides = _int2tuple(strides)
            self._padding = _int2tuple(padding)
            self._kwargs = {
                'kernel_size': self._kernel_size,
                'strides': self._strides,
                'padding': self._padding
            }
            self._input_dtype = input_dtype
            self._weight_dtype = weight_dtype
            self._bias_dtype = bias_dtype

            self.weight = self.params.get('weight', shape=(channels, in_channels//groups, *self._kernel_size),
                                          init=weight_initializer, allow_deferred_init=True)
            self.bias = self.params.get('bias', shape=(channels, ),
                                          init=bias_initializer, allow_deferred_init=True) if use_bias else None
            self.act = nn.Activation(activation, prefix=activation+'_') if activation is not None else None

    def hybrid_forward(self, F, inputs, weight, bias=None):
        # Cast
        inputs = F.cast(inputs, self._input_dtype)
        weight = F.cast(weight, self._weight_dtype)
        bias = F.cast(bias, self._bias_dtype) if bias is not None else None
        # Split inputs, weight and bias according to groups
        inputs = F.split(inputs, self._groups, axis=1) if self._groups > 1 else [inputs]
        weight = F.split(weight, self._groups, axis=0) if self._groups > 1 else [weight]
        bias = F.split(bias, self._groups, axis=0) if self._groups > 1 and bias is not None else [bias]*self._groups
        # Apply conv for each group
        y = []
        for x, w, b in zip(inputs, weight, bias):
            # im2col
            x, out_h, out_w = _im2col_2D(F, x, **self._kwargs)  # batch_size x (out_height * out_width) x (kh * kw * in_channels)
            # Transport weight to matrix
            w = w.reshape((w.shape[0], -1))    # channels x (kh * kw * in_channels)
            # Do convolution operation
            x = F.dot(x, w, transpose_b=True)     # batch_size x (out_height * out_width) x channels
            if b is not None:
                x = x + b.reshape((1, 1, -1))
            # Rearrange output
            x = x.swapaxes(1, 2)    # batch_size x channels x (out_height * out_width)
            y.append(x.reshape((x.shape[0], x.shape[1], out_h, out_w)))
        # Concat feature maps of every group
        return F.concat(*y, dim=1)




