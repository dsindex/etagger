from __future__ import print_function
import tensorflow as tf
import numpy as np
from embvec import EmbVec
from attention import multihead_attention, normalize

class Model:

    __wrd_keep_prob = 0.5          # keep probability for dropout(word embedding)
    __chr_keep_prob = 0.5          # keep probability for dropout(character embedding)
    __rnn_keep_prob = 0.5          # keep probability for dropout(rnn cell)
    __pos_keep_prob = 0.5          # keep probability for dropout(pos embedding)
    __filter_sizes = [3]           # filter sizes
    __num_filters = 30             # number of filters
    __chr_embedding_type = 'conv'  # 'max' | 'conv', default is max
    __rnn_size = 200               # size of RNN hidden unit
    __num_layers = 2               # number of RNN layers
    __mh_num_heads = 4             # number of head for multi head attention
    __mh_num_units = 32            # Q,K,V dimension for multi head attention
    __mh_dropout = 0.5             # dropout probability for multi head attention

    def __init__(self, config):
        self.embvec = config.embvec
        self.sentence_length = config.sentence_length
        self.wrd_vocab_size = len(self.embvec.wrd_embeddings)
        self.wrd_dim = config.wrd_dim
        self.word_length = config.word_length
        self.chr_vocab_size = len(self.embvec.chr_vocab)
        self.chr_dim = config.chr_dim
        self.pos_vocab_size = len(self.embvec.pos_vocab)
        self.pos_dim = config.pos_dim
        self.etc_dim = config.etc_dim
        self.class_size = config.class_size
        self.is_train = config.is_train
        self.use_crf = config.use_crf
        self.set_cuda_visible_devices(self.is_train)

        """
        Input layer
        """
        # (large) word embedding data
        self.wrd_embeddings_init = tf.placeholder(tf.float32, shape=[self.wrd_vocab_size, self.wrd_dim])
        self.wrd_embeddings = tf.Variable(self.wrd_embeddings_init, name='wrd_embeddings', trainable=False)
        # word embedding features
        self.input_data_word_ids = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='input_data_word_ids')
        keep_prob = self.__wrd_keep_prob if self.is_train else 1.0
        self.word_embeddings = self.__word_embedding(self.input_data_word_ids, keep_prob=keep_prob, scope='word-embedding')
        # character embedding features
        self.input_data_wordchr_ids = tf.placeholder(tf.int32, shape=[None, self.sentence_length, self.word_length], name='input_data_wordchr_ids')
        keep_prob = self.__chr_keep_prob if self.is_train else 1.0
        self.wordchr_embeddings = self.__wordchr_embedding(self.input_data_wordchr_ids, keep_prob=keep_prob, scope='wordchr-embedding')

        # pos embedding features
        self.input_data_pos_ids = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='input_data_pos_ids')
        keep_prob = self.__pos_keep_prob if self.is_train else 1.0
        self.pos_embeddings = self.__pos_embedding(self.input_data_pos_ids, keep_prob=keep_prob, scope='pos-embedding')

        # etc features 
        self.input_data_etc = tf.placeholder(tf.float32, shape=[None, self.sentence_length, self.etc_dim], name='input_data_etc')

        self.input_data = tf.concat([self.word_embeddings, self.wordchr_embeddings, self.pos_embeddings, self.input_data_etc], axis=-1, name='input_data') # (batch_size, sentence_length, unit_dim)

        """
        RNN layer
        """
        self.length = self.__compute_length(self.input_data_etc)
        rnn_output = tf.identity(self.input_data)
        for i in range(self.__num_layers):
            keep_prob = self.__rnn_keep_prob if self.is_train else 1.0
            scope = 'bi-lstm-%s' % i
            rnn_output = self.__bi_lstm(rnn_output, self.length, rnn_size=self.__rnn_size, keep_prob=keep_prob, scope=scope)
        self.rnn_output = rnn_output

        """
        Attention layer
        """
        self.attended_output = self.__self_attention(self.rnn_output, 2*self.__rnn_size, scope='self-attention')

        """
        Projection layer
        """
        with tf.variable_scope('projection'):
            weight = tf.Variable(tf.truncated_normal([2*self.__rnn_size, self.class_size], stddev=0.01), name='W')
            bias = tf.Variable(tf.constant(0.1, shape=[self.class_size]), name='b')
            t_attended_output = tf.reshape(self.attended_output, [-1, 2*self.__rnn_size])      # (batch_size*sentence_length, 2*self.__rnn_size)
            self.logits = tf.matmul(t_attended_output, weight) + bias                          # (batch_size*sentence_length, class_size)
            self.logits = tf.reshape(self.logits, [-1, self.sentence_length, self.class_size]) # (batch_size, sentence_length, class_size)
            self.logits_indices = tf.cast(tf.argmax(self.logits, 2), tf.int32)                 # (batch_size, sentence_length)

        """
        Output answer
        """
        self.output_data = tf.placeholder(tf.float32, shape=[None, self.sentence_length, self.class_size], name='output_data')
        self.output_data_indices = tf.cast(tf.argmax(self.output_data, 2), tf.int32)           # (batch_size, sentence_length)

        """
        Loss, Accuracy, Optimization
        """
        with tf.variable_scope('loss'):
            self.loss = self.__compute_loss()

        with tf.variable_scope('accuracy'):
            self.accuracy = self.__compute_accuracy()

        with tf.variable_scope('optimization'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def __word_embedding(self, inputs, keep_prob=0.5, scope='word-embedding'):
        """Look up word embeddings
        """
        with tf.variable_scope(scope):
            with tf.device('/cpu:0'):
                word_embeddings = tf.nn.embedding_lookup(self.wrd_embeddings, inputs)   # (batch_size, sentence_length, wrd_dim)
            return tf.nn.dropout(word_embeddings, keep_prob)

    def __wordchr_embedding(self, inputs, keep_prob=0.5, scope='wordchr-embedding'):
        """Compute character embeddings
        """
        with tf.variable_scope(scope):
            with tf.device('/cpu:0'):
                chr_embeddings = tf.Variable(tf.random_uniform([self.chr_vocab_size, self.chr_dim], -1.0, 1.0), name='chr_embeddings')
                wordchr_embeddings = tf.nn.embedding_lookup(chr_embeddings, inputs)                         # (batch_size, sentence_length, word_length, chr_dim)
                wordchr_embeddings = tf.reshape(wordchr_embeddings, [-1, self.word_length, self.chr_dim])   # (batch_size*sentence_length, word_length, chr_dim)
                if self.__chr_embedding_type == 'conv':
                    wordchr_embeddings = tf.expand_dims(wordchr_embeddings, -1)                             # (batch_size*sentence_length, word_length, chr_dim, 1)
            if self.__chr_embedding_type == 'conv':
                pooled_outputs = []
                for i, filter_size in enumerate(self.__filter_sizes):
                    with tf.variable_scope('conv-maxpool-%s' % filter_size):
                        # convolution layer
                        filter_shape = [filter_size, self.chr_dim, 1, self.__num_filters]
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                        conv = tf.nn.conv2d(
                            wordchr_embeddings,
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
                            ksize=[1, self.word_length - filter_size + 1, 1, 1],
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
                h_pool = tf.concat(pooled_outputs, axis=-1)
                h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
                """
                h_pool Tensor("concat:0", shape=(?, 1, 1, num_filters_total), dtype=float32)
                h_pool_flat Tensor("Reshape:0", shape=(?, num_filters_total), dtype=float32)
                """
                h_drop = tf.nn.dropout(h_pool_flat, keep_prob)
                # (batch_size*sentence_length, num_filters_total) -> (batch_size, sentence_length, num_filters_total)
                wordchr_embeddings = tf.reshape(h_drop, [-1, self.sentence_length, num_filters_total])
            else: # simple character embedding by reduce_max
                wordchr_embeddings = tf.reduce_max(wordchr_embeddings, reduction_indices=1)                   # (batch_size*sentence_length, chr_dim)
                wordchr_embeddings = tf.reshape(wordchr_embeddings, [-1, self.sentence_length, self.chr_dim]) # (batch_size, sentence_length, chr_dim)
                wordchr_embeddings = tf.nn.dropout(wordchr_embeddings, keep_prob)
            return wordchr_embeddings

    def __pos_embedding(self, inputs, keep_prob=0.5, scope='pos-embedding'):
        with tf.variable_scope(scope):
            with tf.device('/cpu:0'):
                p_embeddings = tf.Variable(tf.random_uniform([self.pos_vocab_size, self.pos_dim], -0.5, 0.5), name='p_embeddings')
                pos_embeddings = tf.nn.embedding_lookup(p_embeddings, inputs)    # (batch_size, sentence_length, pos_dim)
            return tf.nn.dropout(pos_embeddings, keep_prob)

    def __bi_lstm(self, inputs, lengths, rnn_size, keep_prob=0.5, scope='bi-lstm'):
        """Apply bi-directional LSTM
        """
        with tf.variable_scope(scope):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size)
            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=lengths, dtype=tf.float32)
            outputs = tf.concat([output_fw, output_bw], axis=-1)
            return tf.nn.dropout(outputs, keep_prob)

    def __self_attention(self, inputs, model_dim, scope='self-attention'):
        """Apply self attention
        """
        with tf.variable_scope(scope):
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

    def __compute_loss(self):
        """Compute loss(self.output_data, self.logits)
        """
        if self.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.output_data_indices, self.length)
            self.trans_params = trans_params # need to evaludate it for decoding
            return tf.reduce_mean(-log_likelihood)
        else:
            cross_entropy = self.output_data * tf.log(tf.nn.softmax(self.logits))        # (batch_size, sentence_length, class_size)
            cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)           # (batch_size, sentence_length)
            # ignore padding by masking
            mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2)) # (batch_size, sentence_length)
            cross_entropy *= mask
            cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)            # (batch_size)
            cross_entropy /= tf.cast(self.length, tf.float32)                            # (batch_size)
            self.trans_params = tf.constant(0.0, shape=[self.class_size, self.class_size])
            return tf.reduce_mean(cross_entropy)

    def __compute_accuracy(self):
        """Compute accuracy(self.output_data, self.logits)
        """
        correct_prediction = tf.cast(tf.equal(self.logits_indices, self.output_data_indices), tf.float32)  # (batch_size, sentence_length)
        # ignore padding by masking
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))                       # (batch_size, sentence_length)
        correct_prediction *= mask
        correct_prediction = tf.reduce_sum(correct_prediction, reduction_indices=1)                        # (batch_size)
        correct_prediction /= tf.cast(self.length, tf.float32)                                             # (batch_size)
        return tf.reduce_mean(correct_prediction)

    def __compute_length(self, output_data):
        """Compute each sentence length
        """
        words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(output_data), reduction_indices=2)) # (batch_size, sentence_length)
        return tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32)      # (batch_size)

    @staticmethod
    def set_cuda_visible_devices(is_train):
        import os
        os.environ['CUDA_VISIBLE_DEVICES']='2'
        if is_train:
            from tensorflow.python.client import device_lib
            print(device_lib.list_local_devices())
