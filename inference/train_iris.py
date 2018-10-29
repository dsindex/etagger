#!./bin/env python

import sys
import tensorflow as tf
import numpy as np

def one_hot(y_data) :
    a = np.array(y_data, dtype=int)
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b

def prepare_data(xy_data):
    x_data = xy_data[1:]
    x_data = np.transpose(x_data)                 # None x 4
    '''
    [ [2    14    33    50],
      [24    56    31    67],
      [23    51    31    69],
      .... ]
    '''
    print(x_data)
    y_data = xy_data[0]                # 1 x None
    '''
    [ [0 1 1 0 1 2 .... ] ]
    '''
    y_data = one_hot(y_data)           # None x 3
    '''
    [ [1 0 0],
    [0 1 0],
    [0 1 0],
    ... ]
    '''
    print(y_data)
    return x_data, y_data

X = tf.placeholder("float", [None, 4], name='X')
Y = tf.placeholder("float", [None, 3], name='Y')
W = tf.Variable(tf.truncated_normal([4,3], stddev=0.01), name='W')
b = tf.Variable(tf.constant(0.1, shape=[3]), name='b')
logits = tf.nn.softmax(tf.matmul(X, W) + b, name='logits')

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(logits), reduction_indices=1))
learning_rate = tf.Variable(0.001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training
xy_data = np.loadtxt('./data/iris.txt', unpack=True, dtype='float32')
with tf.Session() as sess:
    init_all_vars_op = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
    sess.run(init_all_vars_op)
    x_data, y_data = prepare_data(xy_data)
    for i in range(2000):
        if i % 100 == 0 :
            print("step : ", i)
            print("cost : ", sess.run(cost, feed_dict={X: x_data, Y: y_data}))
            print(sess.run(W))
            print("training accuracy :", sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))
        sess.run(train, feed_dict={X:x_data, Y:y_data})

    # save graph and weights
    saver = tf.train.Saver(tf.global_variables())
    checkpoint_dir = './exported'
    checkpoint_file = 'iris_model'
    saver.save(sess, checkpoint_dir + '/' + checkpoint_file)
    tf.train.write_graph(sess.graph, '.', "./exported/graph.pb", as_text=False)
    tf.train.write_graph(sess.graph, '.', "./exported/graph.pb_txt", as_text=True)
    graph = tf.get_default_graph()
    for op in graph.get_operations():
        print(op.name)
    t1 = graph.get_tensor_by_name('logits:0')
    t2 = graph.get_tensor_by_name('W:0')
    t3 = graph.get_tensor_by_name('b:0')

    t1, t2, t2, X = sess.run([t1, t2, t3, X], {X: x_data})

    print(tf.global_variables())
    print('X       ', X)
    print('logits  ', t1)
    print('W       ', t2)
    print('b       ', t3)
    

