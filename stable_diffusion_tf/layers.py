import tensorflow as tf
from tensorflow import keras


class PaddedConv2D(keras.layers.Layer):
    def __init__(self, channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.padding2d = keras.layers.ZeroPadding2D((padding, padding))
        self.conv2d = keras.layers.Conv2D(
            channels, kernel_size, strides=(stride, stride)
        )

    def call(self, x):
        x = self.padding2d(x)
        return self.conv2d(x)


class GEGLU(keras.layers.Layer):
    def __init__(self, dim_out):
        super().__init__()
        self.proj = keras.layers.Dense(dim_out * 2)
        self.dim_out = dim_out

    def call(self, x):
        xp = self.proj(x)
        x, gate = xp[..., : self.dim_out], xp[..., self.dim_out :]
        return x * gelu(gate)


def gelu(x):
    tanh_res = keras.activations.tanh(x * 0.7978845608 * (1 + 0.044715 * (x**2)))
    return 0.5 * x * (1 + tanh_res)


def quick_gelu(x):
    return x * tf.sigmoid(x * 1.702)


def apply_seq(x, layers):
    for l in layers:
        x = l(x)
    return x

