from __future__ import print_function
import tensorflow as tf
import numpy as np
from embvec import EmbVec
from attention import multihead_attention, normalize
from masked_conv import masked_conv1d_and_max

class Model:

    __wrd_keep_prob = 0.5          # keep probability for dropout(word embedding)
    __chr_keep_prob = 0.5          # keep probability for dropout(character embedding)
    __rnn_keep_prob = 0.5          # keep probability for dropout(rnn cell)
    __pos_keep_prob = 0.5          # keep probability for dropout(pos embedding)
    __chr_conv_type = 'conv2d'     # conv1d | conv2d
    __filter_sizes = [3]           # filter sizes
    __num_filters = 30             # number of filters
    __rnn_type = 'fused'           # normal | fused
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
        self.use_crf = config.use_crf
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        _ = tf.cond(self.is_train, lambda: self.set_cuda_visible_devices(True), lambda: self.set_cuda_visible_devices(False)) 

        """
        Input layer
        """
        # (large) word embedding data
        self.wrd_embeddings_init = tf.placeholder(tf.float32, shape=[self.wrd_vocab_size, self.wrd_dim])
        self.wrd_embeddings = tf.Variable(self.wrd_embeddings_init, name='wrd_embeddings', trainable=False)

        # word embedding features
        self.input_data_word_ids = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='input_data_word_ids')
        keep_prob = tf.cond(self.is_train, lambda: self.__wrd_keep_prob, lambda: 1.0)
        self.word_embeddings = self.__word_embedding(self.input_data_word_ids, keep_prob=keep_prob, scope='word-embedding')

        # character embedding features
        self.input_data_wordchr_ids = tf.placeholder(tf.int32, shape=[None, self.sentence_length, self.word_length], name='input_data_wordchr_ids')
        keep_prob = tf.cond(self.is_train, lambda: self.__chr_keep_prob, lambda: 1.0)
        if self.__chr_conv_type == 'conv1d':
            self.wordchr_embeddings = self.__wordchr_embedding_conv1d(self.input_data_wordchr_ids, keep_prob=keep_prob, scope='wordchr-embedding-conv1d')
        else:
            self.wordchr_embeddings = self.__wordchr_embedding_conv2d(self.input_data_wordchr_ids, keep_prob=keep_prob, scope='wordchr-embedding-conv2d')

        # pos embedding features
        self.input_data_pos_ids = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='input_data_pos_ids')
        keep_prob = tf.cond(self.is_train, lambda: self.__pos_keep_prob, lambda: 1.0)
        self.pos_embeddings = self.__pos_embedding(self.input_data_pos_ids, keep_prob=keep_prob, scope='pos-embedding')

        # etc features 
        self.input_data_etcs = tf.placeholder(tf.float32, shape=[None, self.sentence_length, self.etc_dim], name='input_data_etcs')

        self.input_data = tf.concat([self.word_embeddings, self.wordchr_embeddings, self.pos_embeddings, self.input_data_etcs], axis=-1, name='input_data') # (batch_size, sentence_length, unit_dim)
        #self.input_data = tf.concat([self.word_embeddings, self.wordchr_embeddings], axis=-1, name='input_data') # (batch_size, sentence_length, unit_dim)

        """
        RNN layer
        """
        self.sentence_lengths = self.__compute_sentence_lengths(self.input_data_etcs)
        rnn_output = tf.identity(self.input_data)
        for i in range(self.__num_layers):
            keep_prob = tf.cond(self.is_train, lambda: self.__rnn_keep_prob, lambda: 1.0)
            if self.__rnn_type == 'fused':
                scope = 'bi-lstm-fused-%s' % i
                rnn_output = self.__bi_lstm_fused(rnn_output, self.sentence_lengths, rnn_size=self.__rnn_size, keep_prob=keep_prob, scope=scope)
            else:
                scope = 'bi-lstm-%s' % i
                rnn_output = self.__bi_lstm(rnn_output, self.sentence_lengths, rnn_size=self.__rnn_size, keep_prob=keep_prob, scope=scope)
        self.rnn_output = rnn_output

        """
        Attention layer
        """
        self.attended_output = self.__self_attention(self.rnn_output, 2*self.__rnn_size, scope='self-attention')
        #self.attended_output = self.rnn_output

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
            self.learning_rate = tf.train.exponential_decay(config.starter_learning_rate, 
                                                            self.global_step, 
                                                            config.decay_steps, 
                                                            config.decay_rate, 
                                                            staircase=True)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 1)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            '''
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
            '''

    def __word_embedding(self, inputs, keep_prob=0.5, scope='word-embedding'):
        """Look up word embeddings
        """
        with tf.variable_scope(scope):
            with tf.device('/cpu:0'):
                word_embeddings = tf.nn.embedding_lookup(self.wrd_embeddings, inputs)   # (batch_size, sentence_length, wrd_dim)
            return tf.nn.dropout(word_embeddings, keep_prob)

    def __wordchr_embedding_conv1d(self, inputs, keep_prob=0.5, scope='wordchr-embedding-conv1d'):
        """Compute character embeddings by masked conv1d and max-pooling
        """
        with tf.variable_scope(scope):
            with tf.device('/cpu:0'):
                chr_embeddings = tf.Variable(tf.random_uniform([self.chr_vocab_size, self.chr_dim], -1.0, 1.0), name='chr_embeddings')
                wordchr_embeddings_t = tf.nn.embedding_lookup(chr_embeddings, inputs)                           # (batch_size, sentence_length, word_length, chr_dim)
                wordchr_embeddings_t = tf.nn.dropout(wordchr_embeddings_t, keep_prob)
            wordchr_embeddings_t = tf.reshape(wordchr_embeddings_t, [-1, self.word_length, self.chr_dim])       # (batch_size*sentence_length, word_length, chr_dim)
            # masking
            word_masks = self.__compute_word_masks(wordchr_embeddings_t)
            filters = self.__num_filters
            kernel_size = self.__filter_sizes[0]
            wordchr_embeddings = masked_conv1d_and_max(wordchr_embeddings_t, word_masks, filters, kernel_size, tf.nn.relu)
            # (batch_size*sentence_length, filters) -> (batch_size, sentence_length, filters)
            wordchr_embeddings = tf.reshape(wordchr_embeddings, [-1, self.sentence_length, filters])
            return tf.nn.dropout(wordchr_embeddings, keep_prob)

    def __wordchr_embedding_conv2d(self, inputs, keep_prob=0.5, scope='wordchr-embedding-conv2d'):
        """Compute character embeddings by conv2d and max-pooling
        """
        with tf.variable_scope(scope):
            with tf.device('/cpu:0'):
                chr_embeddings = tf.Variable(tf.random_uniform([self.chr_vocab_size, self.chr_dim], -1.0, 1.0), name='chr_embeddings')
                wordchr_embeddings_t = tf.nn.embedding_lookup(chr_embeddings, inputs)                           # (batch_size, sentence_length, word_length, chr_dim)
            wordchr_embeddings_t = tf.reshape(wordchr_embeddings_t, [-1, self.word_length, self.chr_dim])       # (batch_size*sentence_length, word_length, chr_dim)
            # masking
            word_masks = self.__compute_word_masks(wordchr_embeddings_t)
            word_masks = tf.expand_dims(word_masks, -1)                     # (batch_size*sentence_length, word_length, 1)
            wordchr_embeddings_t *= word_masks # broadcasting
            # conv and max-pooling
            wordchr_embeddings = tf.expand_dims(wordchr_embeddings_t, -1)   # (batch_size*sentence_length, word_length, chr_dim, 1)
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
            # (batch_size*sentence_length, num_filters_total) -> (batch_size, sentence_length, num_filters_total)
            wordchr_embeddings = tf.reshape(h_pool_flat, [-1, self.sentence_length, num_filters_total])
            return tf.nn.dropout(wordchr_embeddings, keep_prob)

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

    def __bi_lstm_fused(self, inputs, lengths, rnn_size, keep_prob=0.5, scope='bi-lstm-fused'):
        """Apply bi-directional LSTM fused
        """
        with tf.variable_scope(scope):
            t = tf.transpose(inputs, perm=[1, 0, 2])  # Need time-major
            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(rnn_size)
            lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(rnn_size)
            lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
            output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=lengths)
            output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=lengths)
            outputs = tf.concat([output_fw, output_bw], axis=-1)
            outputs = tf.transpose(outputs, perm=[1, 0, 2])
            return tf.nn.dropout(outputs, keep_prob)

    def __self_attention(self, inputs, model_dim, scope='self-attention'):
        """Apply self attention
        """
        with tf.variable_scope(scope):
            queries = inputs
            keys = inputs
            is_training = tf.cond(self.is_train, lambda: True, lambda: False)
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
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.output_data_indices, self.sentence_lengths)
            self.trans_params = trans_params # need to evaludate it for decoding
            return tf.reduce_mean(-log_likelihood)
        else:
            cross_entropy = self.output_data * tf.log(tf.nn.softmax(self.logits))        # (batch_size, sentence_length, class_size)
            cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)           # (batch_size, sentence_length)
            # masking
            mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2)) # (batch_size, sentence_length)
            cross_entropy *= mask
            cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)            # (batch_size)
            cross_entropy /= tf.cast(self.sentence_lengths, tf.float32)                  # (batch_size)
            self.trans_params = tf.constant(0.0, shape=[self.class_size, self.class_size])
            return tf.reduce_mean(cross_entropy)

    def __compute_accuracy(self):
        """Compute accuracy(self.output_data, self.logits)
        """
        correct_prediction = tf.cast(tf.equal(self.logits_indices, self.output_data_indices), tf.float32)  # (batch_size, sentence_length)
        # masking
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))                       # (batch_size, sentence_length)
        correct_prediction *= mask
        correct_prediction = tf.reduce_sum(correct_prediction, reduction_indices=1)                        # (batch_size)
        correct_prediction /= tf.cast(self.sentence_lengths, tf.float32)                                   # (batch_size)
        return tf.reduce_mean(correct_prediction)

    def __compute_sentence_lengths(self, input_data_etcs):
        """Compute each sentence lengths
        """
        sentence_masks = self.__compute_sentence_masks(input_data_etcs)
        return tf.cast(tf.reduce_sum(sentence_masks, reduction_indices=1), tf.int32)           # (batch_size)

    def __compute_sentence_masks(self, input_data_etcs):
        """Compute each sentence masks
        """
        sentence_masks = tf.sign(tf.reduce_max(tf.abs(input_data_etcs), reduction_indices=2))  # (batch_size, sentence_length, etc_dim) -> (batch_size, sentence_length)
        return sentence_masks

    def __compute_word_lengths(self, wordchr_embeddings_t):
        """Compute each word lengths
        """
        word_masks = self.__compute_word_masks(wordchr_embeddings_t)
        return tf.cast(tf.reduce_sum(word_masks, reduction_indices=1), tf.int32)               # (batch_size*sentence_length)

    def __compute_word_masks(self, wordchr_embeddings_t):
        """Compute each word masks 
        """
        word_masks = tf.sign(tf.reduce_max(tf.abs(wordchr_embeddings_t), reduction_indices=2)) # (batch_size*sentence_length, word_length, chr_dim) -> (batch_size*sentence_length, word_length)
        return word_masks

    @staticmethod
    def set_cuda_visible_devices(is_train):
        import os
        os.environ['CUDA_VISIBLE_DEVICES']='2'
        if is_train:
            from tensorflow.python.client import device_lib
            print(device_lib.list_local_devices())
        return True
