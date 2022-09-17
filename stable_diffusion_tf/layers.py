import tensorflow as tf


def quick_gelu(x):
    return x * tf.sigmoid(x * 1.702) 


def get_conv2d(in_channels, out_channels, kernel_size ,padding=0 ,stride=1):    
    return tf.keras.models.Sequential([
        tf.keras.layers.ZeroPadding2D(
            padding=(padding, padding), data_format=None 
        ),
        tf.keras.layers.Conv2D(
            out_channels,
            kernel_size,
            strides=(stride, stride )) 
    ])
   

def apply_seq(x , layers):
    for l in layers:
        x = l(x)
    return x

def td_dot(a , b ):
    assert len(a.shape) == 4
    assert len(a.shape) == 4
    assert a.shape[0] == b.shape[0]
    assert b.shape[1] == a.shape[1]
    aa = tf.reshape(a , (-1 , a.shape[2] , a.shape[3]))
    bb = tf.reshape(b , (-1 , b.shape[2] , b.shape[3]))
    cc = tf.keras.backend.batch_dot(aa , bb )
    c = tf.reshape(cc , (-1 , a.shape[1] , cc.shape[1] , cc.shape[2]))
    return c 


def gelu(self):
    return 0.5 * self * (1 + tf.keras.activations.tanh(self * 0.7978845608 * (1 + 0.044715 * self * self)))

class GEGLU(tf.keras.layers.Layer):
    def __init__(self, dim_in, dim_out):
        super(GEGLU, self).__init__()
        
        self.proj = tf.keras.layers.Dense( dim_out * 2 )
        self.dim_out = dim_out

    def __call__(self, x):
        xp = self.proj(x)
        x, gate = xp[... , :self.dim_out ] , xp[..., self.dim_out:]
        ans = x * gelu(gate)
        
        return ans
