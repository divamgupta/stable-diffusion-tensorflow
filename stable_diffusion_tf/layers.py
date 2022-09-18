import tensorflow as tf
from tensorflow import keras


def quick_gelu(x):
    return x * tf.sigmoid(x * 1.702)


def apply_seq(x, layers):
    for l in layers:
        x = l(x)
    return x


def td_dot(a, b):
    assert len(a.shape) == 4
    assert len(a.shape) == 4
    assert a.shape[0] == b.shape[0]
    assert b.shape[1] == a.shape[1]
    aa = tf.reshape(a, (-1, a.shape[2], a.shape[3]))
    bb = tf.reshape(b, (-1, b.shape[2], b.shape[3]))
    cc = keras.backend.batch_dot(aa, bb)
    return tf.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))


def gelu(x):
    tanh_res = keras.activations.tanh(x * 0.7978845608 * (1 + 0.044715 * (x**2)))
    return 0.5 * x * (1 + tanh_res)


class PaddedConv2D(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.padding2d = keras.layers.ZeroPadding2D(
            padding=(padding, padding), data_format=None
        )
        self.conv2d = keras.layers.Conv2D(
            out_channels, kernel_size, strides=(stride, stride)
        )

    def call(self, x):
        x = self.padding2d(x)
        return self.conv2d(x)


class GEGLU(keras.layers.Layer):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = keras.layers.Dense(dim_out * 2)
        self.dim_out = dim_out

    def call(self, x):
        xp = self.proj(x)
        x, gate = xp[..., : self.dim_out], xp[..., self.dim_out :]
        return x * gelu(gate)
