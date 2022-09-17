import tensorflow as tf
from .layers import quick_gelu, get_conv2d, apply_seq, td_dot, gelu, GEGLU
import tensorflow_addons as tfa
import numpy as np


class CLIPAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(CLIPAttention, self).__init__()
        self.embed_dim = 768
        self.num_heads = 12
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.k_proj = tf.keras.layers.Dense(self.embed_dim  , activation=None)
        self.v_proj = tf.keras.layers.Dense(self.embed_dim , activation=None)
        self.q_proj = tf.keras.layers.Dense(self.embed_dim , activation=None)
        self.out_proj = tf.keras.layers.Dense(self.embed_dim , activation=None)

    def _shape(self, tensor, seq_len: int, bsz: int):
        a = tf.reshape(tensor , (bsz, seq_len, self.num_heads, self.head_dim))
        return tf.keras.layers.Permute( (2,1,3) )(a) # bs , n_head , seq_len , head_dim
   
    def call(self, inputs ):
        [hidden_states, causal_attention_mask] = inputs
        bsz, tgt_len, embed_dim = hidden_states.shape
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), tgt_len, -1)
        value_states = self._shape(self.v_proj(hidden_states), tgt_len, -1)
        
        proj_shape = (-1 , tgt_len, self.head_dim)
        query_states = self._shape(query_states, tgt_len, -1)
        query_states = tf.reshape(query_states , proj_shape)
        key_states = tf.reshape(key_states , proj_shape)
        
        src_len = tgt_len
        value_states = tf.reshape(value_states , proj_shape)
        attn_weights = query_states @ tf.keras.layers.Permute((2,1))(key_states) 
        
        attn_weights = tf.reshape(attn_weights , (-1, self.num_heads, tgt_len, src_len))
        attn_weights = attn_weights + causal_attention_mask
        attn_weights = tf.reshape(attn_weights , (-1, tgt_len, src_len))
        
        attn_weights = tf.nn.softmax(attn_weights)
        attn_output = attn_weights @ value_states
        
        attn_output = tf.reshape(attn_output , (-1, self.num_heads, tgt_len, self.head_dim))
        attn_output = tf.keras.layers.Permute((2,1,3))(attn_output)
        attn_output = tf.reshape(attn_output , (-1, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)
        return attn_output


class CLIPMLP(tf.keras.layers.Layer):
    def __init__(self):
        super(CLIPMLP, self).__init__()
        self.fc1 = tf.keras.layers.Dense( 3072)
        self.fc2 = tf.keras.layers.Dense(  768)

    def __call__(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = quick_gelu(hidden_states )
        hidden_states = self.fc2(hidden_states)
        return hidden_states



class CLIPEncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CLIPEncoderLayer, self).__init__()
        self.self_attn = CLIPAttention( )
        self.layer_norm1 = tf.keras.layers.LayerNormalization( epsilon=1e-5 )
        self.mlp = CLIPMLP()
        self.layer_norm2 = tf.keras.layers.LayerNormalization( epsilon=1e-5 )

    def __call__(self, inputs):
        [hidden_states, causal_attention_mask] = inputs
        residual = hidden_states
    
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn([hidden_states, causal_attention_mask])
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)        
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super(CLIPEncoder, self).__init__()
        self.layers = [CLIPEncoderLayer() for i in range(12)]
  
    def __call__(self, inputs):
        [hidden_states, causal_attention_mask] = inputs
        for i,l in enumerate(self.layers):
            hidden_states = l([hidden_states, causal_attention_mask])
        return hidden_states



class CLIPTextEmbeddings(tf.keras.layers.Layer):
    def __init__(self , n_words=77):
        super(CLIPTextEmbeddings, self).__init__()

        self.token_embedding_layer = tf.keras.layers.Embedding( 49408, 768 , name="token_embedding" )
        self.position_embedding_layer = tf.keras.layers.Embedding( n_words, 768, name="position_embedding")


    def __call__(self, inputs):
        [input_ids, position_ids] = inputs
        word_embeddings = self.token_embedding_layer(input_ids)
        position_embeddings = self.position_embedding_layer(position_ids)
        return  word_embeddings + position_embeddings


class CLIPTextTransformer(tf.keras.models.Model):
    def __init__(self , n_words=77):
        super(CLIPTextTransformer, self).__init__()
        self.embeddings = CLIPTextEmbeddings( n_words=n_words)
        self.encoder = CLIPEncoder()
        self.final_layer_norm = tf.keras.layers.LayerNormalization( epsilon=1e-5)
    
        # self.inp_position_ids = tf.constant(np.array(list(range(n_words)))[None].astype('int32'))
        self.causal_attention_mask = tf.constant(np.triu(np.ones((1,1,77,77), dtype=np.float32) * -np.inf, k=1))
        
    def __call__(self, inputs, training=False):
        [input_ids, position_ids] = inputs
        x = self.embeddings([input_ids,  position_ids ])
        x = self.encoder([x, self.causal_attention_mask] )
        return self.final_layer_norm(x)

