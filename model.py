from __future__ import print_function
import tensorflow as tf
import numpy as np
from embvec import EmbVec
from attention import multihead_attention, normalize

class Model:

    __rnn_size = 200               # size of RNN hidden unit
    __num_layers = 2               # number of RNN layers
    __cnn_keep_prob = 0.5          # keep probability for dropout(cnn character embedding)
    __rnn_keep_prob = 0.5          # keep probability for dropout(rnn cell)
    __pos_keep_prob = 0.5          # keep probability for dropout(pos embedding)
    __filter_sizes = [3]           # filter sizes
    __num_filters = 30             # number of filters
    __chr_embedding_type = 'conv'  # 'max' | 'conv', default is max
    __mh_num_heads = 4             # number of head for multi head attention
    __mh_num_units = 32            # Q,K,V dimension for multi head attention
    __mh_dropout = 0.5             # dropout probability for multi head attention

    def __init__(self, config):
        embvec = config.embvec
        sentence_length = config.sentence_length
        word_length = config.word_length
        chr_vocab_size = len(embvec.chr_vocab)
        chr_dim = config.chr_dim
        pos_vocab_size = len(embvec.pos_vocab)
        pos_dim = config.pos_dim
        etc_dim = config.etc_dim
        class_size = config.class_size
        self.is_train = config.is_train
        self.set_cuda_visible_devices(self.is_train)

        """
        Input layer
        """
        # word embedding features
        self.input_data_word_ids = tf.placeholder(tf.int32, shape=[None, sentence_length], name='input_data_word_ids')
        with tf.device('/cpu:0'), tf.name_scope('word-embedding'):
            embed_arr = np.array(embvec.wrd_embeddings)
            embed_init = tf.constant_initializer(embed_arr)
            wrd_embeddings = tf.get_variable(name='wrd_embeddings', initializer=embed_init, shape=embed_arr.shape, trainable=False)
            self.word_embeddings = tf.nn.embedding_lookup(wrd_embeddings, self.input_data_word_ids) # (batch_size, sentence_length, wrd_dim)

        # character embedding features
        self.input_data_wordchr_ids = tf.placeholder(tf.int32, shape=[None, sentence_length, word_length], name='input_data_wordchr_ids')
        with tf.name_scope('wordchr-embeddings'):
            with tf.device('/cpu:0'):
                chr_embeddings = tf.Variable(tf.random_uniform([chr_vocab_size, chr_dim], -1.0, 1.0), name='chr_embeddings')
                self.wordchr_embeddings = tf.nn.embedding_lookup(chr_embeddings, self.input_data_wordchr_ids) # (batch_size, sentence_length, word_length, chr_dim)
                self.wordchr_embeddings = tf.reshape(self.wordchr_embeddings, [-1, word_length, chr_dim])     # (batch_size*sentence_length, word_length, chr_dim)
                if self.__chr_embedding_type == 'conv':
                    self.wordchr_embeddings = tf.expand_dims(self.wordchr_embeddings, -1)                     # (batch_size*sentence_length, word_length, chr_dim, 1)
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
                        """ex) for filter size 3 
                        conv Tensor("conv-maxpool-3/conv:0", shape=(?, 13, 1, num_filters), dtype=float32)
                        pooled Tensor("conv-maxpool-3/pool:0", shape=(?, 1, 1, num_filters), dtype=float32)
                        """
                # combine all the pooled features
                num_filters_total = self.__num_filters * len(self.__filter_sizes)
                self.h_pool = tf.concat(pooled_outputs, axis=-1)
                self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
                """
                h_pool Tensor("concat:0", shape=(?, 1, 1, num_filters_total), dtype=float32)
                h_pool_flat Tensor("Reshape:0", shape=(?, num_filters_total), dtype=float32)
                """
                keep_prob = self.__cnn_keep_prob if self.is_train else 1.0
                self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob)
                # (batch_size*sentence_length, num_filters_total) -> (batch_size, sentence_length, num_filters_total)
                self.wordchr_embeddings = tf.reshape(self.h_drop, [-1, sentence_length, num_filters_total])
            else: # simple character embedding by reduce_max
                self.wordchr_embeddings = tf.reduce_max(self.wordchr_embeddings, reduction_indices=1)         # (batch_size*sentence_length, chr_dim)
                self.wordchr_embeddings = tf.reshape(self.wordchr_embeddings, [-1, sentence_length, chr_dim]) # (batch_size, sentence_length, chr_dim)

        # pos embedding features
        self.input_data_pos_ids = tf.placeholder(tf.int32, shape=[None, sentence_length], name='input_data_pos_ids')
        with tf.name_scope('pos-embeddings'):
            with tf.device('/cpu:0'):
                pos_embeddings = tf.Variable(tf.random_uniform([pos_vocab_size, pos_dim], -0.5, 0.5), name='pos_embeddings')
                self.pos_embeddings = tf.nn.embedding_lookup(pos_embeddings, self.input_data_pos_ids) # (batch_size, sentence_length, pos_dim)
                keep_prob = self.__pos_keep_prob if self.is_train else 1.0
                self.pos_embeddings = tf.nn.dropout(self.pos_embeddings, keep_prob)

        # etc features 
        with tf.name_scope('etc'):
            self.input_data_etc = tf.placeholder(tf.float32, shape=[None, sentence_length, etc_dim], name='input_data_etc')

        self.input_data = tf.concat([self.word_embeddings, self.wordchr_embeddings, self.pos_embeddings, self.input_data_etc], axis=-1, name='input_data') # (batch_size, sentence_length, unit_dim)

        """
        RNN layer
        """
        self.length = self.__compute_length(self.input_data_etc)
        with tf.name_scope('rnn'):
            keep_prob = self.__rnn_keep_prob if self.is_train else 1.0
            fw_cell = tf.contrib.rnn.MultiRNNCell([self.__create_cell(self.__rnn_size, keep_prob=keep_prob) for _ in range(self.__num_layers)], state_is_tuple=True)
            bw_cell = tf.contrib.rnn.MultiRNNCell([self.__create_cell(self.__rnn_size, keep_prob=keep_prob) for _ in range(self.__num_layers)], state_is_tuple=True)
            # (batch_size, sentence_length, unit_dim) -> (sentence_length, batch_size, unit_dim) -> list of (batch_size, unit_dim)
            output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell,
                                                   tf.unstack(tf.transpose(self.input_data, perm=[1, 0, 2])),
                                                   dtype=tf.float32, sequence_length=self.length)
            # list of (batch_size, 2*self.__rnn_size) -> (sentence_length, batch_size, 2*self.__rnn_size) -> (batch_size, sentence_length, 2*self.__rnn_size)
            self.rnn_output = tf.transpose(tf.stack(output), perm=[1, 0, 2])

        """
        Attention layer
        """
        self.attended_output = self.__self_attention(self.rnn_output, 2*self.__rnn_size)

        """
        Projection layer
        """
        with tf.name_scope('projection'):
            weight, bias = self.__create_weight_and_bias(2*self.__rnn_size, class_size)
            t_attended_output = tf.reshape(self.attended_output, [-1, 2*self.__rnn_size])    # (batch_size*sentence_length, 2*self.__rnn_size)
            self.prediction = tf.nn.softmax(tf.matmul(t_attended_output, weight) + bias)     # (batch_size*sentence_length, class_size)
            self.prediction = tf.reshape(self.prediction, [-1, sentence_length, class_size]) # (batch_size, sentence_length, class_size)

        """
        Answer
        """
        self.output_data = tf.placeholder(tf.float32, shape=[None, sentence_length, class_size], name='output_data')

        """
        Loss, Accuracy, Optimization
        """
        with tf.name_scope('loss'):
            self.loss = self.__compute_cost()

        with tf.name_scope('accuracy'):
            self.accuracy = self.__compute_accuracy()

        with tf.name_scope('optimization'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


    def __self_attention(self, inputs, model_dim):
        """Apply self attention
        """
        with tf.variable_scope('self-attention'):
            queries = inputs
            keys = inputs
            attended_queries = multihead_attention(queries,
                                                   keys,
                                                   num_units=self.__mh_num_units,
                                                   num_heads=self.__mh_num_heads,
                                                   model_dim=model_dim,
                                                   dropout_rate=self.__mh_dropout,
                                                   is_training=self.is_train,
                                                   causality=False, # no future masking
                                                   scope='multihead-attention',
                                                   reuse=None)
            # residual connection and layer normalization
            return normalize(tf.add(inputs, attended_queries), scope='layer-norm')

    def __compute_cost(self):
        """Compute cross entropy(self.output_data, self.prediction)
        """
        cross_entropy = self.output_data * tf.log(self.prediction)                   # (batch_size, sentence_length, class_size)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)           # (batch_size, sentence_length)
        # ignore padding by masking
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2)) # (batch_size, sentence_length)
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)            # (batch_size)
        cross_entropy /= tf.cast(self.length, tf.float32)                            # (batch_size)
        return tf.reduce_mean(cross_entropy)

    def __compute_accuracy(self):
        """Compute accuracy(self.output_data, self.prediction)
        """
        correct_prediction = tf.cast(tf.equal(tf.argmax(self.prediction, 2), tf.argmax(self.output_data, 2)), 'float') # (batch_size, sentence_length)
        # ignore padding by masking
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2)) # (batch_size, sentence_length)
        correct_prediction *= mask
        correct_prediction = tf.reduce_sum(correct_prediction, reduction_indices=1)  # (batch_size)
        correct_prediction /= tf.cast(self.length, tf.float32)                       # (batch_size)
        return tf.reduce_mean(correct_prediction)

    def __create_cell(self, rnn_size, keep_prob):
        """Create a RNN cell
        """
        cell = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)
        drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return drop

    def __compute_length(self, output_data):
        """Compute each sentence length
        """
        words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(output_data), reduction_indices=2)) # (batch_size, sentence_length)
        length = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32)    # (batch_size)
        return length

    def __create_weight_and_bias(self, in_size, out_size):
        """Create weight matrix and bias
        """
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight, name='projection_weight'), tf.Variable(bias, name='projection_bias')

    @staticmethod
    def set_cuda_visible_devices(is_train):
        import os
        os.environ['CUDA_VISIBLE_DEVICES']='2'
        if is_train:
            from tensorflow.python.client import device_lib
            print(device_lib.list_local_devices())
