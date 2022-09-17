
import tensorflow as tf
import tensorflow_addons as tfa


from .layers import quick_gelu, get_conv2d, apply_seq, td_dot, gelu, GEGLU

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, channels, emb_channels, out_channels):
        super(ResBlock, self).__init__()
        self.in_layers = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            tf.keras.activations.swish,
            get_conv2d( channels, out_channels, 3, padding=1)
        ]
        self.emb_layers = [
            tf.keras.activations.swish,
            tf.keras.layers.Dense( out_channels)
        ]
        self.out_layers = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            tf.keras.activations.swish,
            lambda x: x,
            get_conv2d(out_channels, out_channels, 3, padding=1)
        ]
        self.skip_connection = get_conv2d(channels, out_channels, 1) if channels != out_channels else lambda x: x

    def __call__(self,inputs ):
        x, emb = inputs
        h = apply_seq(x , self.in_layers)
        emb_out = apply_seq(emb, self.emb_layers)
        h = h + emb_out[:, None , None]
        h = apply_seq(h , self.out_layers)
        ret = self.skip_connection(x) + h
        return ret


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super(CrossAttention, self).__init__()
        self.to_q = tf.keras.layers.Dense( n_heads*d_head,use_bias=False )
        self.to_k = tf.keras.layers.Dense( n_heads*d_head,use_bias=False )
        self.to_v = tf.keras.layers.Dense( n_heads*d_head,use_bias=False )
        self.scale = d_head ** -0.5
        self.num_heads = n_heads
        self.head_size = d_head
        self.to_out = [tf.keras.layers.Dense( n_heads*d_head)]

    def __call__(self, inputs):
        assert type(inputs) is list
        if len(inputs) == 1:
                inputs = inputs + [None]
        [x , context] = inputs
        context = x if context is None else context
        q,k,v = self.to_q(x), self.to_k(context), self.to_v(context)
        assert len(x.shape) == 3
        q = tf.reshape(q , ( -1 , x.shape[1], self.num_heads, self.head_size) )
        k = tf.reshape(k , ( -1 , context.shape[1] , self.num_heads, self.head_size) )
        v = tf.reshape(v, ( -1 , context.shape[1] , self.num_heads, self.head_size) )
        
        q = tf.keras.layers.Permute((2,1,3))(q)  # (bs, num_heads, time, head_size)
        k = tf.keras.layers.Permute((2,3,1))(k)  # (bs, num_heads, head_size, time)
        v = tf.keras.layers.Permute((2,1,3))(v)# (bs, num_heads, time, head_size)
        
        score = td_dot(q,k) * self.scale
        weights = tf.keras.activations.softmax(score)                   # (bs, num_heads, time, time)
        attention = td_dot(weights,v)  
        attention = tf.keras.layers.Permute((2,1,3))(attention) # (bs, time, num_heads, head_size)

        h_ = tf.reshape(attention, (-1 ,x.shape[1] , self.num_heads * self.head_size))
        return apply_seq(h_ , self.to_out)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self , dim, mult=4):
        super(FeedForward, self).__init__()
        self.net = [
            GEGLU( dim, dim*mult),
            lambda x: x,
            tf.keras.layers.Dense( dim )
        ]

    def __call__(self, x):
        return apply_seq(x, self.net)


class BasicTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, context_dim, n_heads, d_head):
        super(BasicTransformerBlock, self).__init__()
        self.attn1 = CrossAttention(  dim, dim, n_heads, d_head)
        self.ff = FeedForward(   dim)
        self.attn2 = CrossAttention( dim, context_dim, n_heads, d_head)
        self.norm1 = tf.keras.layers.LayerNormalization( epsilon=1e-5 )
        self.norm2 = tf.keras.layers.LayerNormalization( epsilon=1e-5 )
        self.norm3 = tf.keras.layers.LayerNormalization( epsilon=1e-5 )

    def __call__(self, inputs):
        [x, context] = inputs
        
        x = self.attn1([self.norm1(x)]) + x
        x = self.attn2([self.norm2(x), context]) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(tf.keras.layers.Layer):
    def __init__(self , channels, context_dim, n_heads, d_head):
        super(SpatialTransformer, self).__init__()
        self.norm = tfa.layers.GroupNormalization(epsilon=1e-5 )
        
        assert channels == n_heads * d_head
        self.proj_in = get_conv2d(channels, n_heads * d_head, 1)
        self.transformer_blocks = [BasicTransformerBlock( channels, context_dim, n_heads, d_head)]
        self.proj_out = get_conv2d( n_heads * d_head, channels, 1)

    def __call__(self, inputs):
        [x , context] = inputs
        b, h, w , c  = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = tf.reshape(x , (-1,   h*w , c) ) 
        for block in self.transformer_blocks:
            x = block([x, context])
        x = tf.reshape(x , (-1 , h, w , c) )
        ret = self.proj_out(x) + x_in
        return ret


class Downsample(tf.keras.layers.Layer):
    def __init__(self , channels):
        super(Downsample, self).__init__()
        self.op = get_conv2d( channels, channels, 3, stride=2, padding=1)

    def __call__(self, x):
        return self.op(x)


class Upsample(tf.keras.layers.Layer):
    def __init__(self , channels):
        super(Upsample, self).__init__()
        self.conv = get_conv2d(channels, channels, 3, padding=1)
        self.ups = tf.keras.layers.UpSampling2D(size=(2, 2))

    def __call__(self, x):
        x = self.ups(x)
        return self.conv(x)





class UNetModel(tf.keras.models.Model):
    def __init__(self):
        super(UNetModel, self).__init__()
        self.time_embed = [
             tf.keras.layers.Dense( 1280),
             tf.keras.activations.swish,
             tf.keras.layers.Dense( 1280),
         ]

        self.input_blocks = [
            [get_conv2d(   4, 320, kernel_size=3, padding=1)],
            [ResBlock( 320, 1280, 320) ,   SpatialTransformer( 320, 768, 8, 40)],
            [ResBlock( 320, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
            [Downsample(320)],
            [ResBlock(320, 1280, 640), SpatialTransformer(640, 768, 8, 80)],
            [ResBlock(640, 1280, 640), SpatialTransformer(640, 768, 8, 80)],
            [Downsample(640)],
            [ResBlock(640, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
            [ResBlock(1280, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
            [Downsample(1280)],
            [ResBlock(1280, 1280, 1280)],
            [ResBlock(1280, 1280, 1280)]
        ]
        self.middle_block = [
            ResBlock(1280, 1280, 1280),
            SpatialTransformer(1280, 768, 8, 160),
            ResBlock(1280, 1280, 1280)
        ]
        self.output_blocks = [
            [ResBlock( 2560, 1280, 1280)],
            [ResBlock(2560, 1280, 1280)],
            [ResBlock(2560, 1280, 1280), Upsample(1280)],
            [ResBlock(2560, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
            [ResBlock(2560, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
            [ResBlock(1920, 1280, 1280), SpatialTransformer(1280, 768, 8, 160), Upsample(1280)],
            [ResBlock(1920, 1280, 640), SpatialTransformer(640, 768, 8, 80)],  # 6
            [ResBlock(1280, 1280, 640), SpatialTransformer(640, 768, 8, 80)],
            [ResBlock(960, 1280, 640), SpatialTransformer(640, 768, 8, 80), Upsample(640)],
            [ResBlock(960, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
            [ResBlock(640, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
            [ResBlock(640, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
        ]
        self.out = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
                
            tf.keras.activations.swish,
            get_conv2d( 320, 4, kernel_size=3, padding=1)
        ]

    def __call__(self, inputs, training=False):
        # TODO: real time embedding
        [x,t_emb,context] = inputs
        emb = apply_seq(t_emb, self.time_embed)

        def run(x, bb):
            if isinstance(bb, ResBlock): x = bb([x, emb])
            elif isinstance(bb, SpatialTransformer): x = bb([x, context])
            else: x = bb(x)
            return x

        saved_inputs = []
        for i,b in enumerate(self.input_blocks):
            for bb in b:
                inppp = x
                x = run(x, bb)
            saved_inputs.append(x)
        
        for bb in self.middle_block:
            x = run(x, bb)

        for i,b in enumerate(self.output_blocks):
            x = tf.concat([x , saved_inputs.pop()] , axis=-1) 
            for bb in b:
                x = run(x, bb)
        return apply_seq(x , self.out)


