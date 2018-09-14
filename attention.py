# source from 
# : https://github.com/DongjunLee/transformer-tensorflow/blob/master/transformer/attention.py
# : https://www.programcreek.com/python/example/90548/tensorflow.sequence_mask

from __future__ import print_function
import numpy as np
import tensorflow as tf

__all__ = [
    "positional_encoding", "mask_for_lengths", "Attention"
]

def positional_encoding(dim, sentence_length, dtype=tf.float32):

    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)

def mask_for_lengths(lengths, max_length=None, mask_right=True, value=-1000.0):
    """
    Creates a [batch_size x max_length] mask.
    Args:
        lengths: int32 1-dim tensor of batch_size lengths
        max_length: int32 0-dim tensor or python int
        mask_right: if True, everything before "lengths" becomes zero and the
            rest "value", else vice versa
        value: value for the mask
    Returns:
        [batch_size x max_length] mask of zeros and "value"s
    """
    mask = tf.sequence_mask(lengths, max_length, dtype=tf.float32)
    if mask_right:
        mask = 1.0 - mask
    mask *= value
    return mask 

class Attention:
    """Attention class"""

    def __init__(self,
                 num_heads=1,
                 masked=False,
                 linear_key_dim=50,
                 linear_value_dim=50,
                 model_dim=100,
                 dropout=0.2,
                 lengths=None,
                 max_length=None):

        assert linear_key_dim % num_heads == 0
        assert linear_value_dim % num_heads == 0

        self.num_heads = num_heads
        self.masked = masked
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.dropout = dropout
        self.lengths = lengths
        self.max_length = max_length

    def multi_head(self, q, k, v):
        '''
        # self attention only
        q, k, v = self._mask_for_padding(q, k, v)
        '''
        q, k, v = self._linear_projection(q, k, v)
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs)
        output = self._concat_heads(outputs)
        output = tf.layers.dense(output, self.model_dim)

        return tf.nn.dropout(output, 1.0 - self.dropout)

    def _mask_for_padding(self, q, k, v):
        q_last_dim = tf.shape(q)[-1]
        # [None, max_length]
        mask = mask_for_lengths(self.lengths, self.max_length, mask_right=True, value=-float('inf'))
        # [None, max_length, 1]
        mask = tf.expand_dims(mask, -1)
        # [max, max_length, q_last_dim]
        mask_q = tf.tile(mask, [1, 1, q_last_dim])
        mask_k = mask_q
        mask_v = mask_q
        q = q + mask_q
        k = k + mask_k
        v = v + mask_v
        return q, k, v

    def _linear_projection(self, q, k, v):
        q = tf.layers.dense(q, self.linear_key_dim, use_bias=False)
        k = tf.layers.dense(k, self.linear_key_dim, use_bias=False)
        v = tf.layers.dense(v, self.linear_value_dim, use_bias=False)
        return q, k, v

    def _split_heads(self, q, k, v):

        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, num_heads, max_seq_len, dim]

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.linear_key_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.linear_key_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.linear_value_dim)

        return qs, ks, vs

    def _scaled_dot_product(self, qs, ks, vs):

        key_dim_per_head = self.linear_key_dim // self.num_heads

        o1 = tf.matmul(qs, ks, transpose_b=True)
        o2 = o1 / (key_dim_per_head**0.5)

        if self.masked:
            diag_vals = tf.ones_like(o2[0, 0, :, :]) # (batch_size, num_heads, query_dim, key_dim)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (q_dim, k_dim)
            masks = tf.tile(tf.reshape(tril, [1, 1] + tril.get_shape().as_list()),
                            [tf.shape(o2)[0], tf.shape(o2)[1], 1, 1])
            paddings = tf.ones_like(masks) * -1e9
            o2 = tf.where(tf.equal(masks, 0), paddings, o2)

        o3 = tf.nn.softmax(o2)

        return tf.matmul(o3, vs)

    def _concat_heads(self, outputs):

        def transpose_then_concat_last_two_dimenstion(tensor):
            tensor = tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, max_seq_len, num_heads, dim]
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

        return transpose_then_concat_last_two_dimenstion(outputs)

