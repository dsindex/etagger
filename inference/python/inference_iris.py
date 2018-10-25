#!./bin/env python

import sys
import tensorflow as tf
import numpy as np

'''
X = tf.placeholder("float", [None, 4], name='X')
Y = tf.placeholder("float", [None, 3], name='Y')
W = tf.Variable(tf.truncated_normal([4,3], stddev=0.01), name='W')
b = tf.Variable(tf.constant(0.1, shape=[3]), name='b')
logits = tf.nn.softmax(tf.matmul(X, W) + b, name='logits')
'''

sess = tf.Session()
with sess.as_default():
    checkpoint_dir = './exported'
    checkpoint_file = 'iris_model'
    model_prefix = checkpoint_dir + '/' + checkpoint_file
    # restore meta graph
    meta_file = model_prefix + '.meta'
    loader = tf.train.import_meta_graph(meta_file)
    # mapping init op, placeholders and tensors
    graph = tf.get_default_graph()
    init_all_vars_op = graph.get_operation_by_name('init_all_vars_op')
    W = graph.get_tensor_by_name('W:0')
    b = graph.get_tensor_by_name('b:0')
    X = graph.get_tensor_by_name('X:0')
    logits = graph.get_tensor_by_name('logits:0')
    sess.run(init_all_vars_op)
    # restore actual values
    loader = loader.restore(sess, model_prefix)
    print(tf.global_variables())

    p = sess.run(logits, feed_dict={X:[[2,14,33,50]]}) # 1 0 0 -> type 0
    print(p, sess.run(tf.argmax(p, 1)))

    p = sess.run(logits, feed_dict={X:[[24,56,31,67]]}) # 0 1 0 -> type 1
    print(p, sess.run(tf.argmax(p, 1)))

    p = sess.run(logits, feed_dict={X:[[2,14,33,50], [24,56,31,67]]})
    print(p, sess.run(tf.argmax(p, 1)))
