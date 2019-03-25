#-*- coding: utf-8 -*-

from mxnet.gluon import nn

__all__ = ['Conv2D']
__author__ = "YaHei"

# Helper: convert int to tuple
def _int2tuple(x):
    return (x, ) * 2 if isinstance(x, int) else x

# Helper: im2col operation
def _im2col_2D(F, x, kernel_size, strides):
    # Get parameters
    batch_size, _, height, width = x.shape
    kh, kw = kernel_size
    sh, sw = strides

    # Generate cols
    cols = []
    for h in range(0, height - kh + 1, sh):
        for w in range(0, width - kw + 1, sw):
            # Pick a window, reshape and append
            win = x[:, :, h:(h + kh), w:(w + kw)]
            col = win.reshape((batch_size, -1))     # batch_size x (kh * kw * in_channels)
            cols.append(col)
    # Calculate size of output feature map
    out_h = (height - kh + 1) // sh
    out_w = (width - kw + 1) // sw
    # Stack all cols into a matrix
    return F.stack(*cols, axis=1), out_h, out_w    # batch_size x (out_height * out_width) x (kh * kw * in_channels)

def _quantize(F, x, min_range, max_range):
    x = x.clip(min_range, max_range)
    if max_range == -min_range:
        scale = max_range / 127
    else:
        scale = (max_range - min_range) / 255
    x = F.round(x / scale)
    return F.cast(x, 'int32'), scale

def quantize(F, x, out_type='int8'):
    if out_type == 'int8':
        max_range = x.abs().max().asscalar()
        min_range = -max_range
    elif out_type == 'uint8':
        max_range = x.max().asscalar()
        min_range = x.min().asscalar()
    else:
        raise ValueError("unknown out type: ", out_type)
    return _quantize(F, x, min_range, max_range)

def dequantize(F, x, scale):
    x = F.cast(x, "float32")
    return x * scale

class Conv2D(nn.HybridBlock):
    def __init__(self, channels, kernel_size, strides, padding, in_channels, groups=1,
                 activation=None, use_bias=True, quantized=False,
                 input_dtype='float32', weight_dtype='float32',
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
                'strides': self._strides
            }
            self._quantized = quantized
            self._input_dtype = input_dtype
            self._weight_dtype = weight_dtype
            self._input_range = None
            self._weight_range = None

            self.weight = self.params.get('weight', shape=(channels, in_channels//groups, *self._kernel_size),
                                          init=weight_initializer, allow_deferred_init=True)
            self.bias = self.params.get('bias', shape=(channels, ),
                                          init=bias_initializer, allow_deferred_init=True) if use_bias else None
            self.act = nn.Activation(activation, prefix=activation+'_') if activation is not None else None

    def hybrid_forward(self, F, inputs, weight, bias=None):
        # Pad(note that F.pad is only support float32)
        ph, pw = self._padding
        inputs = inputs.pad(mode='constant', constant_value=0, pad_width=(0, 0, 0, 0, ph, ph, pw, pw))
        # Quantize and cast into int32
        if self._quantized:
            if self._input_range is None:
                inputs, in_scale = quantize(F, inputs, self._input_dtype)
            else:
                inputs, in_scale = _quantize(F, inputs, *self._input_range)
            if self._weight_range is None:
                weight, w_scale = quantize(F, weight, self._weight_dtype)
            else:
                weight, w_scale = _quantize(F, weight, *self._weight_range)
            if bias is not None:
                b_scale = in_scale * w_scale
                b_max = in_scale * w_scale * (2 ** 31)
                bias = F.clip(bias, -b_max, b_max)
                bias = F.round(bias / b_scale)
                bias = F.cast(bias, "int32")
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
            # Do convolution operation(note that F.dot only support float32)
            if self._quantized:
                x_fp = F.cast(x, 'float32')
                w_fp = F.cast(w, 'float32')
                x = F.dot(x_fp, w_fp, transpose_b=True)     # batch_size x (out_height * out_width) x channels
                x = F.cast(x, 'int32')
            else:
                x = F.dot(x, w, transpose_b=True)  # batch_size x (out_height * out_width) x channels
            if b is not None:
                x = x + b.reshape((1, 1, -1))
            # Rearrange output
            x = x.swapaxes(1, 2)    # batch_size x channels x (out_height * out_width)
            y.append(x.reshape((x.shape[0], x.shape[1], out_h, out_w)))
        # Concat feature maps of every group and apply activation
        y = F.concat(*y, dim=1)
        if self.act is not None:
            y = self.act(y)
        # Dequantize
        if self._quantized:
            y = dequantize(F, y, in_scale * w_scale)
        return y

    def __repr__(self):
        s = '{name}({mapping}, kernel_size={kernel_size}, stride={strides}'
        len_kernel_size = len(self._kwargs['kernel_size'])
        if self._padding != (0,) * len_kernel_size:
            s += ', padding={padding}'
        if self._groups != 1:
            s += ', groups={}'.format(self._groups)
        if self.bias is None:
            s += ', bias=False'
        if self.act:
            s += ', {}'.format(self.act)
        s += ')'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        mapping='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]),
                        **self._kwargs)
