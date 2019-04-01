from __future__ import print_function
import tensorflow as tf
import numpy as np
from six.moves import reduce

'''
source from https://github.com/mkroutikov/tf-lstm-char-cnn/blob/master/model.py

input_cnn : [batch_size x num_unroll_steps, cnn_size]
input_cnn = highway(input_cnn, input_cnn.get_shape()[-1], num_layers=num_highway_layers)
'''

def linear(input_, output_size, scope=None):
    """Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]

    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.compat.v1.variable_scope(scope or "SimpleLinear"):
        matrix = tf.compat.v1.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype, use_resource=False)
        bias_term = tf.compat.v1.get_variable("Bias", [output_size], dtype=input_.dtype, use_resource=False)

    return tf.matmul(input_, tf.transpose(a=matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.compat.v1.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output

'''
source from https://github.com/guillaumegenthial/tf_ner/blob/master/models/chars_conv_lstm_crf/masked_conv.py
'''

def masked_conv1d_and_max(t, weights, filters, kernel_size, activation=tf.nn.relu):
    """Applies 1d convolution and a masked max-pooling

    Parameters
    ----------
    t : tf.Tensor
        A tensor with at least 3 dimensions [d1, d2, ..., dn-2, dn-1, dn]
    weights : tf.Tensor of tf.bool
        A Tensor of shape [d1, d2, dn-1]
    filters : int
        number of filters
    kernel_size : int
        kernel size for the temporal convolution
    activation : function
        activation function, ex) tf.nn.relu

    Returns
    -------
    tf.Tensor
        A tensor of shape [d1, d2, ..., dn-2, filters]

    """
    # Get shape and parameters
    shape = tf.shape(input=t)
    ndims = t.shape.ndims
    dim1 = reduce(lambda x, y: x*y, [shape[i] for i in range(ndims - 2)])
    dim2 = shape[-2]
    dim3 = t.shape[-1]

    # Reshape weights
    weights = tf.reshape(weights, shape=[dim1, dim2, 1])
    weights = tf.cast(weights, dtype=tf.float32)

    # Reshape input and apply weights
    flat_shape = [dim1, dim2, dim3]
    t = tf.reshape(t, shape=flat_shape)
    t *= weights

    # Apply convolution
    t_conv = tf.compat.v1.layers.conv1d(t, filters, kernel_size, padding='same', activation=activation)  # (dim1, dim2, filters)
    t_conv *= weights

    # Reduce max -- set to zero if all padded
    t_conv += (1. - weights) * tf.reduce_min(input_tensor=t_conv, axis=-2, keepdims=True)  # (dim1, dim2, filters) + (dim1, 1, filters)
    t_max = tf.reduce_max(input_tensor=t_conv, axis=-2)  # (dim1, 1, filters)

    # Reshape the output
    final_shape = [shape[i] for i in range(ndims-2)] + [filters]
    t_max = tf.reshape(t_max, shape=final_shape) # (d1, d2, .., dn-2, filters)

    return t_max


'''
source from https://github.com/Kyubyong/transformer/blob/master/modules.py
'''

def multihead_attention(queries, 
                        keys, 
                        num_units=32, 
                        num_heads=4,
                        model_dim=400,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    """Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size. (C)
      num_heads: An int. Number of heads. (h)
      model_dim: output model dimension for the last linear projection. (M)
      dropout_rate: A floating point number.
      is_training: Boolean or A bool tensor, Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with shape of (N, T_q, M)  
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        
        # Linear projections
        Q = tf.compat.v1.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.compat.v1.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.compat.v1.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(a=K_, perm=[0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(input_tensor=keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(input=queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(input=outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(input_tensor=queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(input=keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, T_k)
          
        # Dropouts
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(value=is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)

        # Linear projection
        outputs = tf.compat.v1.layers.dense(outputs, model_dim, activation=tf.nn.relu) # (N, T_q, M)
              
    return outputs

def feedforward(inputs,
                masks,
                num_units=[1600, 400],
                kernel_size=1,
                scope="feed-forward", 
                reuse=None):
    """Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      masks: A 2d tensor with shape of [N, T], dtype is tf.float32.
      num_units: A list of two integers.
      kernel_size: A integer value kernel size for conv1d
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Inner layer
        inputs *= masks
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": kernel_size,
                  "padding": "same", "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs *= masks
        
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": kernel_size,
                  "padding": "same", "activation": None, "use_bias": True}
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs *= masks
    
    return outputs

def normalize(inputs, 
              epsilon = 1e-8,
              scope="layer-norm",
              reuse=None):
    """Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(x=inputs, axes=[-1], keepdims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def positional_encoding(lengths,
                        maxlen,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    """Sinusoidal Positional_Encoding.

    Args:
      lengths: The lengths of the inputs to create position embeddings for.
        An int32 tensor of shape `[batch_size]`.
      maxlen: The maximum length of the input sequence to create position
        embeddings for. An int32 tensor.
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor of shape `[batch_size, maxlen, num_units]` that contains
      embeddings for each position. All elements past `lengths` are zero.
    """

    N = tf.shape(input=lengths)[0]
    T = maxlen
    Limit = 1024 # FIXME trick because we can't use range(T)

    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (batch_size, maxlen)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(Limit)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(value=position_enc, dtype=tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(params=lookup_table, ids=position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        # Mask out positions that are padded
        mask = tf.sequence_mask(lengths=lengths, maxlen=maxlen, dtype=tf.float32)
        outputs = outputs * tf.expand_dims(mask, 2) # broadcasting

        return outputs
