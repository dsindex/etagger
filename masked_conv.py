# source from
# : https://github.com/guillaumegenthial/tf_ner/blob/master/models/chars_conv_lstm_crf/masked_conv.py

"""Implement masked 1d convolution with max-pooling"""

__author__ = "Guillaume Genthial"

from six.moves import reduce
import tensorflow as tf


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
    shape = tf.shape(t)
    ndims = t.shape.ndims
    dim1 = reduce(lambda x, y: x*y, [shape[i] for i in range(ndims - 2)])
    dim2 = shape[-2]
    dim3 = t.shape[-1]

    # Reshape weights
    weights = tf.reshape(weights, shape=[dim1, dim2, 1])
    weights = tf.to_float(weights)

    # Reshape input and apply weights
    flat_shape = [dim1, dim2, dim3]
    t = tf.reshape(t, shape=flat_shape)
    t *= weights

    # Apply convolution
    t_conv = tf.layers.conv1d(t, filters, kernel_size, padding='same', activation=activation)  # (dim1, dim2, filters)
    t_conv *= weights

    # Reduce max -- set to zero if all padded
    t_conv += (1. - weights) * tf.reduce_min(t_conv, axis=-2, keep_dims=True)  # (dim1, dim2, filters) + (dim1, 1, filters)
    t_max = tf.reduce_max(t_conv, axis=-2)  # (dim1, 1, filters)

    # Reshape the output
    final_shape = [shape[i] for i in range(ndims-2)] + [filters]
    t_max = tf.reshape(t_max, shape=final_shape) # (d1, d2, .., dn-2, filters)

    return t_max
