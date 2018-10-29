#!./bin/env python

import sys
import tensorflow as tf
import numpy as np

def load_graph(frozen_graph_filename, prefix='prefix'):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            op_dict=None, 
            producer_op_list=None,
            name=prefix,
        )
        
    return graph

frozen_graph_filename = './exported/frozen_model.pb'
graph = load_graph(frozen_graph_filename, prefix='prefix')
for op in graph.get_operations():
    print(op.name)   

W = graph.get_tensor_by_name('prefix/W:0')
b = graph.get_tensor_by_name('prefix/b:0')
X = graph.get_tensor_by_name('prefix/X:0')
logits = graph.get_tensor_by_name('prefix/logits:0')


with tf.Session(graph=graph) as sess:
    print(tf.global_variables())

    p = sess.run(logits, feed_dict={X:[[2,14,33,50]]}) # 1 0 0 -> type 0
    print(p, sess.run(tf.argmax(p, 1)))

    p = sess.run(logits, feed_dict={X:[[24,56,31,67]]}) # 0 1 0 -> type 1
    print(p, sess.run(tf.argmax(p, 1)))

    p = sess.run(logits, feed_dict={X:[[2,14,33,50], [24,56,31,67]]})
    print(p, sess.run(tf.argmax(p, 1)))
