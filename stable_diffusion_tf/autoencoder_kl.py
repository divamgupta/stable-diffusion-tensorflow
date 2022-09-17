
import tensorflow as tf
import tensorflow_addons as tfa

from .layers import quick_gelu, get_conv2d, apply_seq, td_dot, gelu, GEGLU



class AttnBlock(tf.keras.layers.Layer):
    def __init__(self , in_channels):
        super(AttnBlock, self).__init__()
        self.norm = tfa.layers.GroupNormalization(epsilon=1e-5 )
        self.q = get_conv2d(in_channels, in_channels, 1)
        self.k = get_conv2d( in_channels, in_channels, 1)
        self.v = get_conv2d( in_channels, in_channels, 1)
        self.proj_out = get_conv2d(in_channels, in_channels, 1)

    # copied from AttnBlock in ldm repo
    def __call__(self, x):
        h_ = self.norm(x)
        q,k,v = self.q(h_), self.k(h_), self.v(h_)

        # compute attention
        b, h,w, c = q.shape
        q = tf.reshape(q , (-1 ,h*w , c ))# b,hw,c
        k = tf.keras.layers.Permute((3,1,2))(k)
        k = tf.reshape(k , (-1,c,h*w)) # b,c,hw
        w_ = q @ k
        w_ = w_ * (c**(-0.5))
        w_ = tf.keras.activations.softmax (w_)
        


        # attend to values
        v = tf.keras.layers.Permute((3,1,2))(v)
        v = tf.reshape(v , (-1,c,h*w))
        w_ = tf.keras.layers.Permute((2,1))( w_)
        h_ = v @ w_
        h_ = tf.keras.layers.Permute((2,1))(h_)
        h_ = tf.reshape(h_, (-1,h,w,c))
        return x + self.proj_out(h_)



class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self ,  in_channels, out_channels=None):
        super(ResnetBlock, self).__init__()
        self.norm1 = tfa.layers.GroupNormalization(epsilon=1e-5 )
        self.conv1 = get_conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = tfa.layers.GroupNormalization(epsilon=1e-5 )
        self.conv2 = get_conv2d(out_channels, out_channels, 3, padding=1)
        self.nin_shortcut = get_conv2d(in_channels, out_channels, 1) if in_channels != out_channels else lambda x: x

    def __call__(self, x):
        h = self.conv1(tf.keras.activations.swish(self.norm1(x)) )
        h = self.conv2(tf.keras.activations.swish(self.norm2(h)) )
        return self.nin_shortcut(x) + h




class Mid(tf.keras.layers.Layer):
    def __init__(self , block_in):
        super(Mid, self).__init__()
        self.block_1 = ResnetBlock( block_in, block_in)
        self.attn_1 = AttnBlock(block_in)
        self.block_2 = ResnetBlock( block_in, block_in)

    def __call__(self, x):
        return apply_seq(x , [self.block_1, self.attn_1, self.block_2])




class Decoder(tf.keras.models.Model):
    def __init__(self ):
        super(Decoder, self).__init__()
        sz = [(128, 256), (256, 512), (512, 512), (512, 512)]
        self.conv_in = get_conv2d(  4,512,3, padding=1)
        self.mid = Mid( 512)
        self.upp = tf.keras.layers.UpSampling2D(size=(2, 2))
        
        self.post_quant_conv = get_conv2d( 4, 4, 1)

        arr = []
        for i,s in enumerate(sz):
            arr.append({"block":
                [ResnetBlock(  s[1], s[0]),
                 ResnetBlock(  s[0], s[0]),
                 ResnetBlock( s[0], s[0])]})
            if i != 0: arr[-1]['upsample'] = {"conv": get_conv2d(s[0], s[0], 3, padding=1)}
        self.up = arr

        self.norm_out = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.conv_out = get_conv2d(128, 3, 3, padding=1)

    def __call__(self, x, training=False):
        
        x = self.post_quant_conv(1/0.18215 * x)

        x = self.conv_in(x)
        x = self.mid(x)

        for l in self.up[::-1]:
            for b in l['block']: x = b(x)
            if 'upsample' in l:
                # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html ?
                bs,c,py,px = x.shape
                x = self.upp(x)
                x = l['upsample']['conv'](x)

        return self.conv_out(tf.keras.activations.swish (self.norm_out(x)) )

