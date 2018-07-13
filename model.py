from __future__ import print_function
import tensorflow as tf
import numpy as np
from embvec import EmbVec

class Model:
    '''
    RNN model for sequence tagging
    '''

    __rnn_size = 256          # size of RNN hidden unit
    __num_layers = 2          # number of RNN layers
    __keep_prob = 0.5         # keep probability for dropout
    __learning_rate = 0.003   # learning rate

    def __init__(self, config):
        '''
        Initialize RNN model
        '''
        embvec = config.embvec
        sentence_length = config.sentence_length
        etc_dim = config.etc_dim
        class_size = config.class_size
        is_train = config.is_train

        # Input layer and Output(answer)
        self.input_data_word_ids = tf.placeholder(tf.int32, [None, sentence_length], name='input_data_word_dis')
        embed_arr = np.array(embvec.embeddings)
        embed_init = tf.constant_initializer(embed_arr)
        embeddings = tf.get_variable(name='embeddings', initializer=embed_init, shape=embed_arr.shape, trainable=False)
        # embedding_lookup([None, sentence_length]) -> [None, sentence_length, emb_dim]
        self.word_embeddings = tf.nn.embedding_lookup(embeddings, self.input_data_word_ids, name='word_embeddings')
        self.input_data_etc = tf.placeholder(tf.float32, [None, sentence_length, etc_dim], name='input_data_etc')
        # concat([None, sentence_length, emb_dim], [None, sentence_length, etc_dim]) -> [None, sentence_length, word_dim]
        self.input_data = tf.concat([self.word_embeddings, self.input_data_etc], axis=-1, name='input_data')
        self.output_data = tf.placeholder(tf.float32, [None, sentence_length, class_size], name='output_data')

        # RNN layer
        if is_train == 'train':
            keep_prob = self.__keep_prob
        else:
            # do not apply dropout for inference 
            keep_prob = 1.0
        fw_cell = tf.contrib.rnn.MultiRNNCell([self.create_cell(self.__rnn_size, keep_prob=keep_prob) for _ in range(self.__num_layers)], state_is_tuple=True)
        bw_cell = tf.contrib.rnn.MultiRNNCell([self.create_cell(self.__rnn_size, keep_prob=keep_prob) for _ in range(self.__num_layers)], state_is_tuple=True)
        self.length = self.compute_length(self.input_data)
        # transpose([None, sentence_length, word_dim]) -> unstack([sentence_length, None, word_dim]) -> list of [None, word_dim]
        output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell,
                                               tf.unstack(tf.transpose(self.input_data, perm=[1, 0, 2])),
                                               dtype=tf.float32, sequence_length=self.length)
        # stack(list of [None, 2*self.__rnn_size]) -> transpose([sentence_length, None, 2*self.__rnn_size]) -> reshpae([None, sentence_length, 2*self.__rnn_size]) -> [None, 2*self.__rnn_size]
        output = tf.reshape(tf.transpose(tf.stack(output), perm=[1, 0, 2]), [-1, 2*self.__rnn_size])

        # Projection layer
        weight, bias = self.create_weight_and_bias(2*self.__rnn_size, class_size)
        # [None, 2*self.__rnn_size] x [2*self.__rnn_size, class_size] + [class_size]  -> softmax([None, class_size]) -> [None, class_size]
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        # reshape([None, class_size]) -> [None, sentence_length, class_size]
        self.prediction = tf.reshape(prediction, [-1, sentence_length, class_size])

        # Loss and Optimization
        self.loss = self.compute_cost()
        optimizer = tf.train.AdamOptimizer(self.__learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def compute_cost(self):
        '''
        Compute cross entropy(self.output_data, self.prediction)
        '''
        # [None, sentence_length, class_size] * log([None, sentence_length, class_size]) -> [None, sentence_length, class_size]
        # reduce_sum([None, sentence_length, class_size]) -> [None, sentence_length] = [ [0.8, 0.2, ..., 0], [0, 0.7, 0.3, ..., 0], ... ]
        cross_entropy = self.output_data * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        # reduce_max(abs([None, sentence_length, class_size])) -> sign([None, sentence_length]) = [ [1, 0, 0, ..., 0], [0, 1, 1, ..., 0], ... ]
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

    @staticmethod
    def create_cell(rnn_size, keep_prob):
        '''
        Create a RNN cell
        '''
        cell = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)
        drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return drop

    @staticmethod
    def compute_length(input_data):
        '''
        Compute each sentence length in input_data
        '''
        # reduce_max(abs([None, sentence_length, word_dim])) -> sign([None, sentence_length]) = [ [1, 1, 1, ..., 0], [1, 1, 1, ..., 0], ... ] 
        # reduce_sum([None, sentence_length]) -> [None] = [11, 16, 13, ..., 123] (batch_size)
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
        return tf.Variable(weight, name='projection_weight'), tf.Variable(bias, name='projection_bias')

