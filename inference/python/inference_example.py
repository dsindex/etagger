import tensorflow as tf
import numpy as np

val = np.array([[1, 1]], dtype=np.float32)

with tf.Session() as sess:

    # restore meta graph
    loader = tf.train.import_meta_graph('./exported/my_model.meta')
    # mapping placeholders and tensors
    x = tf.get_default_graph().get_tensor_by_name('input:0')
    output = tf.get_default_graph().get_tensor_by_name('output:0')
    kernel = tf.get_default_graph().get_tensor_by_name('dense/kernel:0')
    bias = tf.get_default_graph().get_tensor_by_name('dense/bias:0')
    # restore actual values
    loader = loader.restore(sess, './exported/my_model')

    x, output, kernel, bias = sess.run([x, output, kernel, bias], {x: val})

    print(tf.global_variables())
    print("input           ", x)
    print("output          ", output)
    print("dense/kernel:0  ", kernel)
    print("dense/bias:0    ", bias)
