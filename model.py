from __future__ import print_function
import tensorflow as tf
import numpy as np
from embvec import EmbVec

class Model:
    '''
    RNN model for sequence tagging
    '''

    __rnn_size = 256               # size of RNN hidden unit
    __num_layers = 2               # number of RNN layers
    __cnn_keep_prob = 0.5          # keep probability for dropout(cnn)
    __rnn_keep_prob = 0.5          # keep probability for dropout(rnn)
    __learning_rate = 0.001        # learning rate
    __filter_sizes = [3,4,5]       # filter sizes
    __num_filters = 32             # number of filters
    __chr_embedding_type = 'conv'  # 'max' | 'conv', default is max

    def __init__(self, config):
        '''
        Initialize RNN model
        '''
        embvec = config.embvec
        sentence_length = config.sentence_length
        word_length = config.word_length
        chr_vocab_size = len(embvec.chr_vocab)
        chr_dim = config.chr_dim
        etc_dim = config.etc_dim
        class_size = config.class_size
        is_train = config.is_train
        self.set_cuda_visible_devices(is_train)

        # Input layer

        self.input_data_word_ids = tf.placeholder(tf.int32, shape=[None, sentence_length], name='input_data_word_ids')
        # word embedding features
        with tf.device('/cpu:0'), tf.name_scope('word-embedding'):
            embed_arr = np.array(embvec.wrd_embeddings)
            embed_init = tf.constant_initializer(embed_arr)
            wrd_embeddings = tf.get_variable(name='wrd_embeddings', initializer=embed_init, shape=embed_arr.shape, trainable=False)
            # embedding_lookup([None, sentence_length]) -> [None, sentence_length, wrd_dim]
            self.word_embeddings = tf.nn.embedding_lookup(wrd_embeddings, self.input_data_word_ids, name='word_embeddings')

        # character embedding features
        self.input_data_wordchr_ids = tf.placeholder(tf.int32, shape=[None, sentence_length, word_length], name='input_data_wordchr_ids')
        with tf.name_scope('wordchr-embeddings'):
            with tf.device('/cpu:0'):
                chr_embeddings = tf.Variable(tf.random_uniform([chr_vocab_size, chr_dim], -1.0, 1.0), name='chr_embeddings')
                # embedding_lookup([None, sentence_length, word_length]) -> [None, sentence_length, word_length, chr_dim]
                self.wordchr_embeddings = tf.nn.embedding_lookup(chr_embeddings, self.input_data_wordchr_ids, name='wordchr_embeddings')
                # reshape([None, sentence_length, word_length, chr_dim]) -> [None, word_length, chr_dim]
                self.wordchr_embeddings = tf.reshape(self.wordchr_embeddings, [-1, word_length, chr_dim])
                if self.__chr_embedding_type == 'conv':
                    # expaned_dims([None, word_length, chr_dim]) -> [None, word_length, chr_dim, 1]
                    self.wordchr_embeddings = tf.expand_dims(self.wordchr_embeddings, -1)
            if self.__chr_embedding_type == 'conv':
                pooled_outputs = []
                for i, filter_size in enumerate(self.__filter_sizes):
                    with tf.name_scope('conv-maxpool-%s' % filter_size):
                        # convolution layer
                        filter_shape = [filter_size, chr_dim, 1, self.__num_filters]
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                        conv = tf.nn.conv2d(
                            self.wordchr_embeddings,
                            W,
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name='conv')
                        # apply nonlinearity
                        b = tf.Variable(tf.constant(0.1, shape=[self.__num_filters]), name='b')
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                        # max-pooling over the outputs
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, word_length - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name='pool')
                        pooled_outputs.append(pooled)
                        ''' for filter size 3 
                        conv Tensor("conv-maxpool-3/conv:0", shape=(?, 13, 1, num_filters), dtype=float32)
                        pooled Tensor("conv-maxpool-3/pool:0", shape=(?, 1, 1, num_filters), dtype=float32)
                        '''
                # combine all the pooled features
                num_filters_total = self.__num_filters * len(self.__filter_sizes)
                self.h_pool = tf.concat(pooled_outputs, axis=-1)
                self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
                '''
                h_pool Tensor("concat:0", shape=(?, 1, 1, num_filters_total), dtype=float32)
                h_pool_flat Tensor("Reshape:0", shape=(?, num_filters_total), dtype=float32)
                '''
                if is_train: keep_prob = self.__cnn_keep_prob
                else: keep_prob = 1.0 # do not apply dropout for inference
                self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob)
                # reshape([-1, num_filters_total]) -> [None, sentence_length, num_filters_total]
                self.wordchr_embeddings = tf.reshape(self.h_drop, [-1, sentence_length, num_filters_total])
            else: # simple character embedding by reduce_max
                # reduce_max([None, word_length, chr_dim]) -> [None, chr_dim]
                self.wordchr_embeddings = tf.reduce_max(self.wordchr_embeddings, reduction_indices=1)
                # reshape([None, chr_dim]) -> [None, sentence_length, chr_dim]
                self.wordchr_embeddings = tf.reshape(self.wordchr_embeddings, [-1, sentence_length, chr_dim])

        with tf.name_scope('etc'):
            # etc features 
            self.input_data_etc = tf.placeholder(tf.float32, shape=[None, sentence_length, etc_dim], name='input_data_etc')

        # concat([None, sentence_length, wrd_dim], [None, sentence_length, chr_dim], [None, sentence_length, etc_dim]) -> [None, sentence_length, unit_dim]
        self.input_data = tf.concat([self.word_embeddings, self.wordchr_embeddings, self.input_data_etc], axis=-1, name='input_data')

        # Answer

        self.output_data = tf.placeholder(tf.float32, shape=[None, sentence_length, class_size], name='output_data')

        # RNN layer

        with tf.name_scope('rnn'):
            if is_train: keep_prob = self.__rnn_keep_prob
            else: keep_prob = 1.0 # do not apply dropout for inference 
            fw_cell = tf.contrib.rnn.MultiRNNCell([self.create_cell(self.__rnn_size, keep_prob=keep_prob) for _ in range(self.__num_layers)], state_is_tuple=True)
            bw_cell = tf.contrib.rnn.MultiRNNCell([self.create_cell(self.__rnn_size, keep_prob=keep_prob) for _ in range(self.__num_layers)], state_is_tuple=True)
            self.length = self.compute_length(self.output_data)
            # transpose([None, sentence_length, unit_dim]) -> unstack([sentence_length, None, unit_dim]) -> list of [None, unit_dim]
            output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell,
                                                   tf.unstack(tf.transpose(self.input_data, perm=[1, 0, 2])),
                                                   dtype=tf.float32, sequence_length=self.length)
            # stack(list of [None, 2*self.__rnn_size]) -> transpose([sentence_length, None, 2*self.__rnn_size]) -> reshpae([None, sentence_length, 2*self.__rnn_size]) -> [None, 2*self.__rnn_size]
            output = tf.reshape(tf.transpose(tf.stack(output), perm=[1, 0, 2]), [-1, 2*self.__rnn_size])

        # Projection layer

        with tf.name_scope('projection'):
            weight, bias = self.create_weight_and_bias(2*self.__rnn_size, class_size)
            # [None, 2*self.__rnn_size] x [2*self.__rnn_size, class_size] + [class_size]  -> softmax([None, class_size]) -> [None, class_size]
            prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
            # reshape([None, class_size]) -> [None, sentence_length, class_size]
            self.prediction = tf.reshape(prediction, [-1, sentence_length, class_size])

        # Loss, Accuracy, Optimization

        with tf.name_scope('loss'):
            self.loss = self.compute_cost()

        with tf.name_scope('accuracy'):
            self.accuracy = self.compute_accuracy()

        with tf.name_scope('optimization'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(self.__learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def compute_cost(self):
        '''
        Compute cross entropy(self.output_data, self.prediction)
        '''
        # [None, sentence_length, class_size] * log([None, sentence_length, class_size]) -> [None, sentence_length, class_size]
        # reduce_sum([None, sentence_length, class_size]) -> [None, sentence_length] = [ [0.8, 0.2, ..., 0], [0, 0.7, 0.3, ..., 0], ... ]
        cross_entropy = self.output_data * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        # ignore padding by masking
        # reduce_max(abs([None, sentence_length, class_size])) -> sign([None, sentence_length]) = [ [1, 1, 1,..., 0,..., 0], [1, 1, 1,...,0,..., 0], ... ]
        # [None, sentence_length] * [None, sentence_length] -> [None, sentence_length] (masked)
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))
        cross_entropy *= mask
        # reduce_sum([None, sentence_length]) -> [None] = [2.9, 3.6, 0.4, 0, ... , 0] (batch_size)
        # cast([None], tf.float32) -> [11.0, 16.0, 13.0, ..., 123.0]
        # [None] / [None] -> [None]
        # reduce_mean([None]) -> scalar
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    def compute_accuracy(self):
        # argmax([None, sentence_length, class_size]) -> equal([None, sentence_length]) -> cast([None, sentence_length]) -> [None, sentence_length]
        correct_prediction = tf.cast(tf.equal(tf.argmax(self.prediction, 2), tf.argmax(self.output_data, 2)), 'float')
        # ignore padding by masking
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))
        correct_prediction *= mask
        correct_prediction = tf.reduce_sum(correct_prediction, reduction_indices=1)
        correct_prediction /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(correct_prediction)

    @staticmethod
    def create_cell(rnn_size, keep_prob):
        '''
        Create a RNN cell
        '''
        cell = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)
        drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return drop

    @staticmethod
    def compute_length(output_data):
        '''
        Compute each sentence length
        '''
        # reduce_max(abs([None, sentence_length, dim])) -> sign([None, sentence_length]) = [ [1, 1, 1, ..., 0], [1, 1, 1, ..., 0], ... ] 
        # reduce_sum([None, sentence_length]) -> [None] = [11, 16, 13, ..., 123] (batch_size)
        words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(output_data), reduction_indices=2))
        length = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32)
        return length

    @staticmethod
    def create_weight_and_bias(in_size, out_size):
        '''
        Create weight matrix and bias
        '''
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight, name='projection_weight'), tf.Variable(bias, name='projection_bias')

    @staticmethod
    def set_cuda_visible_devices(is_train):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        if is_train:
            from tensorflow.python.client import device_lib
            print(device_lib.list_local_devices())
