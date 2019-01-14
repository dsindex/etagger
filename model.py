from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
from embvec import EmbVec
from transformer import multihead_attention, feedforward, normalize, positional_encoding
from masked_conv import masked_conv1d_and_max

class Model:

    __keep_prob = 0.7              # keep probability for dropout
    __chr_conv_type = 'conv1d'     # conv1d | conv2d
    __filter_sizes = [3]           # filter sizes
    __num_filters = 25             # number of filters
    __rnn_used = True              # use rnn layer or not
    __rnn_num_layers = 2           # number of RNN layers
    __rnn_type = 'fused'           # normal | fused
    __rnn_size = 200               # size of RNN hidden unit
    __tf_used = False              # use transformer encoder layer or not
    __tf_num_layers = 4            # number of layers for transformer encoder
    __tf_keep_prob = 0.8           # keep probability for transformer encoder
    __tf_mh_num_heads = 4          # number of head for multi head attention
    __tf_mh_num_units = 64         # Q,K,V dimension for multi head attention
    __tf_mh_keep_prob = 0.8        # keep probability for multi head attention
    __tf_ffn_kernel_size = 3       # conv1d kernel size for feed forward net
    __tf_ffn_keep_prob = 0.8       # keep probability for feed forward net

    def __init__(self, config):
        self.embvec = config.embvec
        self.wrd_vocab_size = len(self.embvec.wrd_embeddings)
        self.wrd_dim = config.wrd_dim
        self.word_length = config.word_length
        self.chr_vocab_size = len(self.embvec.chr_vocab)
        self.chr_dim = config.chr_dim
        self.pos_vocab_size = len(self.embvec.pos_vocab)
        self.pos_dim = config.pos_dim
        self.class_size = config.class_size
        self.use_crf = config.use_crf
        self.emb_class = config.emb_class
        self.set_cuda_visible_devices(config.arg_train)

        """
        Input layer
        """
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.sentence_length = tf.placeholder(tf.int32, name='sentence_length')
        keep_prob = tf.cond(self.is_train, lambda: self.__keep_prob, lambda: 1.0)

        # pos embedding
        self.input_data_pos_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_data_pos_ids') # (batch_size, sentence_length)
        self.sentence_masks   = self.__compute_sentence_masks(self.input_data_pos_ids)
        sentence_lengths = self.__compute_sentence_lengths(self.sentence_masks)
        self.sentence_lengths = tf.identity(sentence_lengths, name='sentence_lengths')
        masks = tf.to_float(tf.expand_dims(self.sentence_masks, -1))
        self.pos_embeddings = self.__pos_embedding(self.input_data_pos_ids, keep_prob=keep_prob, scope='pos-embedding')

        # (large) word embedding data
        self.wrd_embeddings_init = tf.placeholder(tf.float32, shape=[self.wrd_vocab_size, self.wrd_dim], name='wrd_embeddings_init')
        self.wrd_embeddings = tf.Variable(self.wrd_embeddings_init, name='wrd_embeddings', trainable=False)
        # word embeddings
        self.input_data_word_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_data_word_ids') # (batch_size, sentence_length)
        self.word_embeddings = self.__word_embedding(self.input_data_word_ids, keep_prob=keep_prob, scope='word-embedding')

        # character embeddings
        self.input_data_wordchr_ids = tf.placeholder(tf.int32,
                                                     shape=[None, None, self.word_length], # (batch_size, sentence_length, word_length)
                                                     name='input_data_wordchr_ids')
        if self.__chr_conv_type == 'conv1d':
            self.wordchr_embeddings = self.__wordchr_embedding_conv1d(self.input_data_wordchr_ids,
                                                                      keep_prob=keep_prob,
                                                                      scope='wordchr-embedding-conv1d')
        else:
            self.wordchr_embeddings = self.__wordchr_embedding_conv2d(self.input_data_wordchr_ids,
                                                                      keep_prob=keep_prob,
                                                                      scope='wordchr-embedding-conv2d')

        if self.emb_class == 'elmo':
            # elmo embeddings
            self.elmo_bilm = config.elmo_bilm
            self.elmo_input_data_wordchr_ids = tf.placeholder(tf.int32,
                                                              shape=[None, None, self.word_length], # (batch_size, sentence_length+2, word_length)
                                                              name='elmo_input_data_wordchr_ids')   # '+2' stands for '<S>', '</S>'
            self.elmo_embeddings = self.__elmo_embedding(self.elmo_input_data_wordchr_ids, masks, keep_prob=keep_prob)

        if self.emb_class == 'elmo':
            self.input_data = tf.concat([self.word_embeddings, self.wordchr_embeddings, self.elmo_embeddings, self.pos_embeddings],
                                        axis=-1,
                                        name='input_data') # (batch_size, sentence_length, input_dim)
        else:
            self.input_data = tf.concat([self.word_embeddings, self.wordchr_embeddings, self.pos_embeddings],
                                        axis=-1,
                                        name='input_data') # (batch_size, sentence_length, input_dim)
        # masking (for confirmation)
        self.input_data *= masks

        """
        RNN layer
        """
        rnn_output = tf.identity(self.input_data)
        if self.__rnn_used:
            for i in range(self.__rnn_num_layers):
                if self.__rnn_type == 'fused':
                    scope = 'bi-lstm-fused-%s' % i
                    rnn_output = self.__bi_lstm_fused(rnn_output,
                                                      self.sentence_lengths,
                                                      rnn_size=self.__rnn_size,
                                                      keep_prob=keep_prob,
                                                      scope=scope)                # (batch_size, sentence_length, 2*rnn_size)
                else:
                    scope = 'bi-lstm-%s' % i
                    rnn_output = self.__bi_lstm(rnn_output,
                                                self.sentence_lengths,
                                                rnn_size=self.__rnn_size,
                                                keep_prob=keep_prob, scope=scope) # (batch_size, sentence_length, 2*rnn_size)
        self.rnn_output = rnn_output

        """
        Transformer layer
        """
        transformed_output = tf.identity(self.rnn_output)
        if self.__tf_used:
            tf_keep_prob = tf.cond(self.is_train, lambda: self.__tf_keep_prob, lambda: 1.0)
            tf_mh_keep_prob = tf.cond(self.is_train, lambda: self.__tf_mh_keep_prob, lambda: 1.0)
            tf_ffn_keep_prob = tf.cond(self.is_train, lambda: self.__tf_ffn_keep_prob, lambda: 1.0)
            # last dimension must be equal to model_dim because we use a residual connection. 
            model_dim = transformed_output.get_shape().as_list()[-1]
            # sinusoidal positional signal
            signal = positional_encoding(self.sentence_lengths,
                                         self.sentence_length,
                                         transformed_output.get_shape().as_list()[-1],
                                         zero_pad=False,
                                         scale=False,
                                         scope='positional-encoding',
                                         reuse=None)
            transformed_output += signal
            # block
            for i in range(self.__tf_num_layers):
                x = transformed_output
                # layer norm
                x_norm = normalize(x, scope='layer-norm-sa-%s'%i, reuse=None)
                # multi-head attention
                y = self.__self_attention(x_norm,
                                          masks,
                                          model_dim=model_dim,
                                          keep_prob=tf_mh_keep_prob,
                                          scope='self-attention-%s'%i)
                # residual and dropout
                x = tf.nn.dropout(x_norm + y, keep_prob=tf_keep_prob)
                # layer norm
                x_norm = normalize(x, scope='layer-norm-ffn-%s'%i, reuse=None)
                # position-wise feed forward net
                y = self.__feedforward(x_norm,
                                       masks,
                                       model_dim=model_dim,
                                       kernel_size=self.__tf_ffn_kernel_size,
                                       keep_prob=tf_ffn_keep_prob,
                                       scope='feed-forward-%s'%i)
                # residual and dropout
                x = tf.nn.dropout(x_norm + y, keep_prob=tf_keep_prob)
                transformed_output = x
            # final layer norm
            transformed_output = normalize(transformed_output, scope='layer-norm', reuse=None)
        self.transformed_output = transformed_output

        """
        Projection layer
        """
        logits = self.__projection(self.transformed_output,
                                   self.class_size,
                                   scope='projection')                     # (batch_size, sentence_length, class_size)
        self.logits = tf.identity(logits, name='logits')
        logits_indices = tf.cast(tf.argmax(logits, 2), tf.int32)           # (batch_size, sentence_length)
        self.logits_indices = tf.identity(logits_indices, name='logits_indices')

        """
        Output answer
        """
        self.output_data = tf.placeholder(tf.float32,
                                          shape=[None, None, self.class_size], # (batch_size, sentence_length, class_size)
                                          name='output_data')
        self.output_data_indices = tf.cast(tf.argmax(self.output_data, 2), tf.int32)  # (batch_size, sentence_length)

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
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            '''
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
            '''
    
    def __elmo_embedding(self, inputs, masks, keep_prob=0.5):
        """Compute ELMO embeddings
        """
        from bilm import weight_layers
        elmo_embeddings_op = self.elmo_bilm(inputs)
        elmo_input = weight_layers('input', elmo_embeddings_op, l2_coef=0.0)
        elmo_embeddings = elmo_input['weighted_op'] # (batch_size, sentence_length, elmo_dim)
        # masking(remove noise due to padding)
        elmo_embeddings *= masks
        return tf.nn.dropout(elmo_embeddings, keep_prob)

    def __word_embedding(self, inputs, keep_prob=0.5, scope='word-embedding'):
        """Look up word embeddings
        """
        with tf.variable_scope(scope):
            with tf.device('/cpu:0'):
                word_embeddings = tf.nn.embedding_lookup(self.wrd_embeddings, inputs) # (batch_size, sentence_length, wrd_dim)
            return tf.nn.dropout(word_embeddings, keep_prob)

    def __wordchr_embedding_conv1d(self, inputs, keep_prob=0.5, scope='wordchr-embedding-conv1d'):
        """Compute character embeddings by masked conv1d and max-pooling
        """
        with tf.variable_scope(scope):
            with tf.device('/cpu:0'):
                chr_embeddings = tf.Variable(tf.random_uniform([self.chr_vocab_size, self.chr_dim], -1.0, 1.0),
                                             name='chr_embeddings')
                wordchr_embeddings_t = tf.nn.embedding_lookup(chr_embeddings, inputs) # (batch_size, sentence_length, word_length, chr_dim)
                wordchr_embeddings_t = tf.nn.dropout(wordchr_embeddings_t, keep_prob)
            wordchr_embeddings_t = tf.reshape(wordchr_embeddings_t,
                                              [-1, self.word_length, self.chr_dim])   # (batch_size*sentence_length, word_length, chr_dim)
            # masking
            t = tf.reshape(inputs, [-1, self.word_length])  # (batch_size*sentence_length, word_length) 
            masks = self.__compute_word_masks(t)            # (batch_size*sentence_length, word_length)
            filters = self.__num_filters
            kernel_size = self.__filter_sizes[0]
            wordchr_embeddings = masked_conv1d_and_max(wordchr_embeddings_t, masks, filters, kernel_size, tf.nn.relu)
            # (batch_size*sentence_length, filters) -> (batch_size, sentence_length, filters)
            wordchr_embeddings = tf.reshape(wordchr_embeddings, [-1, self.sentence_length, filters])
            return tf.nn.dropout(wordchr_embeddings, keep_prob)

    def __wordchr_embedding_conv2d(self, inputs, keep_prob=0.5, scope='wordchr-embedding-conv2d'):
        """Compute character embeddings by conv2d and max-pooling
        """
        with tf.variable_scope(scope):
            with tf.device('/cpu:0'):
                chr_embeddings = tf.Variable(tf.random_uniform([self.chr_vocab_size, self.chr_dim], -1.0, 1.0),
                                              name='chr_embeddings')
                wordchr_embeddings_t = tf.nn.embedding_lookup(chr_embeddings, inputs) # (batch_size, sentence_length, word_length, chr_dim)
            wordchr_embeddings_t = tf.reshape(wordchr_embeddings_t,
                                              [-1, self.word_length, self.chr_dim])   # (batch_size*sentence_length, word_length, chr_dim)
            # masking
            t = tf.reshape(inputs, [-1, self.word_length])   # (batch_size*sentence_length, word_length) 
            masks = self.__compute_word_masks(t)             # (batch_size*sentence_length, word_length)
            masks = tf.expand_dims(masks, -1)                # (batch_size*sentence_length, word_length, 1)
            wordchr_embeddings_t *= tf.to_float(masks)       # broadcasting
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
        """Computing pos embeddings
        """
        with tf.variable_scope(scope):
            with tf.device('/cpu:0'):
                p_embeddings = tf.Variable(tf.random_uniform([self.pos_vocab_size, self.pos_dim], -0.5, 0.5),
                                           name='p_embeddings')
                pos_embeddings = tf.nn.embedding_lookup(p_embeddings, inputs) # (batch_size, sentence_length, pos_dim)
            # masking
            masks = tf.expand_dims(self.sentence_masks, -1)                   # (batch_size, sentence_length, 1)
            pos_embeddings *= tf.to_float(masks)                              # broadcasting
            return tf.nn.dropout(pos_embeddings, keep_prob)

    def __bi_lstm(self, inputs, lengths, rnn_size, keep_prob=0.5, scope='bi-lstm'):
        """Apply bi-directional LSTM
        """
        with tf.variable_scope(scope):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size)
            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                        cell_bw,
                                                                        inputs,
                                                                        sequence_length=lengths,
                                                                        dtype=tf.float32)
            outputs = tf.concat([output_fw, output_bw], axis=-1)
            return tf.nn.dropout(outputs, keep_prob)

    def __bi_lstm_fused(self, inputs, lengths, rnn_size, keep_prob=0.5, scope='bi-lstm-fused'):
        """Apply bi-directional LSTM block fused
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

    def __self_attention(self, inputs, masks, model_dim=None, keep_prob=0.5, scope='self-attention'):
        """Apply self attention 
        """
        with tf.variable_scope(scope):
            inputs *= masks # inputs should be masked before multihead_attention()
            if not model_dim: model_dim = inputs.get_shape().as_list()[-1]
            queries = inputs
            keys = inputs
            is_training = tf.cond(self.is_train, lambda: True, lambda: False)
            attended_queries = multihead_attention(queries,
                                                   keys,
                                                   num_units=self.__tf_mh_num_units,
                                                   num_heads=self.__tf_mh_num_heads,
                                                   model_dim=model_dim,
                                                   dropout_rate=1.0 - keep_prob,
                                                   is_training=is_training,
                                                   causality=False, # no future masking
                                                   scope='multihead-attention',
                                                   reuse=None)
            return attended_queries

    def __feedforward(self, inputs, masks, model_dim=None, kernel_size=1, keep_prob=0.5, scope='feed-forward'):
        """Apply Point-wise feed forward layer
        """
        with tf.variable_scope(scope):
            if not model_dim: model_dim = inputs.get_shape().as_list()[-1]
            num_units = [4*model_dim, model_dim]
            outputs = feedforward(inputs, masks, num_units=num_units, kernel_size=kernel_size, scope=scope, reuse=None)
            outputs = tf.nn.dropout(outputs, keep_prob)
            return outputs

    def __projection(self, inputs, out_dim, scope='projection'):
        """Apply fully-connected projection layer
        """
        with tf.variable_scope('projection'):
            in_dim = inputs.get_shape().as_list()[-1]
            weight = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=0.01), name='W')
            bias = tf.Variable(tf.constant(0.1, shape=[out_dim]), name='b')
            t_output = tf.reshape(inputs, [-1, in_dim])                      # (batch_size*sentence_length, in_dim)
            output = tf.matmul(t_output, weight) + bias                      # (batch_size*sentence_length, out_dim)
            output = tf.reshape(output, [-1, self.sentence_length, out_dim]) # (batch_size, sentence_length, out_dim)
            return output

    def __compute_loss(self):
        """Compute loss(self.output_data, self.logits)
        """
        trans_params = tf.get_variable('trans_params',
                                       shape=[self.class_size, self.class_size],
                                       initializer=initializers.xavier_initializer())
        self.trans_params = trans_params
        if self.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(inputs=self.logits,
                                                                             tag_indices=self.output_data_indices,
                                                                             transition_params=trans_params,
                                                                             sequence_lengths=self.sentence_lengths)
            return tf.reduce_mean(-log_likelihood)
        else:
            cross_entropy = self.output_data * tf.log(tf.nn.softmax(self.logits)) # (batch_size, sentence_length, class_size)
            cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)    # (batch_size, sentence_length)
            # masking
            masks = self.sentence_masks 
            cross_entropy *= tf.to_float(masks)
            cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)     # (batch_size)
            cross_entropy /= tf.cast(self.sentence_lengths, tf.float32)           # (batch_size)
            trans_params = tf.constant(0.0, shape=[self.class_size, self.class_size])
            return tf.reduce_mean(cross_entropy)

    def __compute_accuracy(self):
        """Compute accuracy(self.output_data, self.logits)
        """
        correct_prediction = tf.cast(tf.equal(self.logits_indices, self.output_data_indices),
                                     tf.float32)                                     # (batch_size, sentence_length)
        # masking
        masks = self.sentence_masks
        correct_prediction *= tf.to_float(masks)
        correct_prediction = tf.reduce_sum(correct_prediction, reduction_indices=1)  # (batch_size)
        correct_prediction /= tf.cast(self.sentence_lengths, tf.float32)             # (batch_size)
        return tf.reduce_mean(correct_prediction)

    def __compute_sentence_lengths(self, sentence_masks):
        """Compute each sentence lengths
        """
        return tf.cast(tf.reduce_sum(sentence_masks, reduction_indices=1), tf.int32) # (batch_size)

    def __compute_sentence_masks(self, t):
        """Compute each sentence masks
        """
        sentence_masks = tf.sign(tf.abs(t)) # (batch_size, sentence_length)
        return sentence_masks

    def __compute_word_masks(self, t):
        """Compute each word masks 
        """
        word_masks = tf.sign(tf.abs(t))    # (batch_size*sentence_length, word_length)
        return word_masks

    @staticmethod
    def set_cuda_visible_devices(arg_train):
        import os
        os.environ['CUDA_VISIBLE_DEVICES']='3'
        if arg_train:
            from tensorflow.python.client import device_lib
            print(device_lib.list_local_devices())
        return True
