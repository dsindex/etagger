from __future__ import print_function
import tensorflow as tf
import numpy as np

class Model:
    '''
    LSTM model for sequence tagging
    '''

    __keep_prob = 0.5
    __learning_rate = 0.003

    def __init__(self, args):
        '''
        Initialize LSTM model
        '''
        self.args = args

        # Input layer and Output(answer)
        self.input_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.word_dim])
        self.output_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.class_size])

        # RNN layer
        fw_cell = tf.contrib.rnn.MultiRNNCell([self.create_cell(args.rnn_size, keep_prob=self.__keep_prob) for _ in range(args.num_layers)], state_is_tuple=True)
        bw_cell = tf.contrib.rnn.MultiRNNCell([self.create_cell(args.rnn_size, keep_prob=self.__keep_prob) for _ in range(args.num_layers)], state_is_tuple=True)
        self.length = self.compute_length(self.input_data)
        # transpose([None, args.sentence_length, args.word_dim]) -> unstack([args.sentence_length, None, args.word_dim]) -> list of [None, args.word_dim]
        output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell,
                                               tf.unstack(tf.transpose(self.input_data, perm=[1, 0, 2])),
                                               dtype=tf.float32, sequence_length=self.length)
        # stack(list of [None, 2*args.rnn_size]) -> transpose([args.sentence_length, None, 2*args.rnn_size]) -> reshpae([None, args.sentence_length, 2*args.rnn_size]) -> [None, 2*args.rnn_size]
        output = tf.reshape(tf.transpose(tf.stack(output), perm=[1, 0, 2]), [-1, 2*args.rnn_size])

        # Fully Connected and Softmax Output layer
        weight, bias = self.create_weight_and_bias(2*args.rnn_size, args.class_size)
        # [None, 2*args.rnn_size] x [2*args.rnn_size, args.class_size] + [args.class_size]  -> softmax([None, args.class_size]) -> [None, args.class_size]
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        # reshape([None, args.class_size]) -> [None, args.sentence_length, args.class_size]
        self.prediction = tf.reshape(prediction, [-1, args.sentence_length, args.class_size])

        # Loss and Optimization
        self.loss = self.cost()
        optimizer = tf.train.AdamOptimizer(self.__learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    @staticmethod
    def cost(self):
        '''
        Compute cross entropy(self.output_data, self.prediction)
        '''
        # [None, args.sentence_length, args.class_size] * log([None, args.sentence_length, args.class_size]) -> [None, args.sentence_length, args.class_size]
        # reduce_sum([None, args.sentence_length, args.class_size]) -> [None, args.sentence_length] = [ [0.8, 0.2, ..., 0], [0, 0.7, 0.3, ..., 0], ... ]
        cross_entropy = self.output_data * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        # reduce_max(abs([None, args.sentence_length, args.class_size])) -> sign([None, args.sentence_length]) = [ [1, 0, 0, ..., 0], [0, 1, 1, ..., 0], ... ]
        # [None, args.sentence_length] * [None, args.sentence_length] -> [None, args.sentence_length] (masked)
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))
        cross_entropy *= mask
        # reduce_sum([None, args.sentence_length]) -> [None] = [2.9, 3.6, 0.4, 0, ... , 0] (args.batch_size)
        # cast([None], tf.float32) -> [11.0, 16.0, 13.0, ..., 123.0]
        # [None] / [None] -> [None]
        # reduce_mean([None]) -> scalar
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @staticmethod
    def create_cell(rnn_size, keep_prob):
        '''
        Create a LSTM cell
        '''
        cell = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)
        drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return drop

    @staticmethod
    def compute_length(input_data):
        '''
        Compute each sentence length in input_data
        '''
        # reduce_max(abs([None, args.sentence_length, args.word_dim])) -> sign([None, args.sentence_length]) = [ [1, 1, 1, ..., 0], [1, 1, 1, ..., 0], ... ] 
        # reduce_sum([None, args.sentence_length]) -> [None] = [11, 16, 13, ..., 123] (args.batch_size)
        words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(input_data), reduction_indices=2))
        length = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32)
        return length

    @staticmethod
    def create_weight_and_bias(in_size, out_size):
        '''
        Create weight matrix and bias
        '''
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


