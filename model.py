from __future__ import print_function
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import tf_metrics
from embvec import EmbVec
from ops import multihead_attention, feedforward, normalize, positional_encoding, masked_conv1d_and_max, highway

class Model:

    def __init__(self, config):
        self.config = config
        self.embvec = config.embvec
        self.wrd_vocab_size = len(self.embvec.wrd_embeddings)
        self.wrd_dim = config.wrd_dim
        self.word_length = config.word_length
        self.chr_vocab_size = len(self.embvec.chr_vocab)
        self.chr_dim = config.chr_dim
        self.pos_vocab_size = len(self.embvec.pos_vocab)
        self.pos_dim = config.pos_dim
        self.chk_vocab_size = len(self.embvec.chk_vocab)
        self.chk_dim = config.chk_dim
        self.class_size = config.class_size
        self.use_crf = config.use_crf
        self.emb_class = config.emb_class
        self.is_training = config.is_training
        self.set_cuda_visible_devices(self.is_training)

        """
        Input layer
        """
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')
        self.sentence_length = tf.compat.v1.placeholder(tf.int32, name='sentence_length')
        keep_prob = tf.cond(pred=self.is_train, true_fn=lambda: config.keep_prob, false_fn=lambda: 1.0)

        # pos embedding
        self.input_data_pos_ids = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='input_data_pos_ids') # (batch_size, sentence_length)
        self.sentence_masks   = self.__compute_sentence_masks(self.input_data_pos_ids)
        sentence_lengths = self.__compute_sentence_lengths(self.sentence_masks)
        self.sentence_lengths = tf.identity(sentence_lengths, name='sentence_lengths')
        masks = tf.cast(tf.expand_dims(self.sentence_masks, -1), dtype=tf.float32)
        self.pos_embeddings = self.__pos_embedding(self.input_data_pos_ids, keep_prob=keep_prob, scope='pos-embedding')

        # chk embedding
        self.input_data_chk_ids = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='input_data_chk_ids') # (batch_size, sentence_length)
        self.chk_embeddings = self.__chk_embedding(self.input_data_chk_ids, keep_prob=keep_prob, scope='chk-embedding')

        # (large) word embedding data
        self.wrd_embeddings_init = tf.compat.v1.placeholder(tf.float32, shape=[self.wrd_vocab_size, self.wrd_dim], name='wrd_embeddings_init')
        self.wrd_embeddings = tf.Variable(self.wrd_embeddings_init, name='wrd_embeddings', trainable=False)
        # word embeddings
        self.input_data_word_ids = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='input_data_word_ids') # (batch_size, sentence_length)
        self.word_embeddings = self.__word_embedding(self.input_data_word_ids, keep_prob=keep_prob, scope='word-embedding')

        # character embeddings
        self.input_data_wordchr_ids = tf.compat.v1.placeholder(tf.int32,
                                                     shape=[None, None, self.word_length], # (batch_size, sentence_length, word_length)
                                                     name='input_data_wordchr_ids')
        if config.chr_conv_type == 'conv1d':
            self.wordchr_embeddings = self.__wordchr_embedding_conv1d(self.input_data_wordchr_ids,
                                                                      keep_prob=keep_prob,
                                                                      scope='wordchr-embedding-conv1d')
        else:
            self.wordchr_embeddings = self.__wordchr_embedding_conv2d(self.input_data_wordchr_ids,
                                                                      keep_prob=keep_prob,
                                                                      scope='wordchr-embedding-conv2d')

        if 'elmo' in self.emb_class:
            # elmo embeddings
            self.elmo_bilm = config.elmo_bilm
            elmo_keep_prob = tf.cond(pred=self.is_train, true_fn=lambda: config.elmo_keep_prob, false_fn=lambda: 1.0)
            self.elmo_input_data_wordchr_ids = tf.compat.v1.placeholder(tf.int32,
                                                              shape=[None, None, self.word_length], # (batch_size, sentence_length+2, word_length)
                                                              name='elmo_input_data_wordchr_ids')   # '+2' stands for '<S>', '</S>'
            self.elmo_embeddings = self.__elmo_embedding(self.elmo_input_data_wordchr_ids, masks, keep_prob=elmo_keep_prob)
        if 'bert' in self.emb_class:
            # bert embeddings
            self.bert_config = config.bert_config
            self.bert_init_checkpoint = config.bert_init_checkpoint
            bert_keep_prob = tf.cond(pred=self.is_train, true_fn=lambda: config.bert_keep_prob, false_fn=lambda: 1.0)
            self.bert_input_data_token_ids   = tf.compat.v1.placeholder(tf.int32, shape=[None, config.bert_max_seq_length], name='bert_input_data_token_ids')
            self.bert_input_data_token_masks = tf.compat.v1.placeholder(tf.int32, shape=[None, config.bert_max_seq_length], name='bert_input_data_token_masks') 
            self.bert_input_data_segment_ids = tf.compat.v1.placeholder(tf.int32, shape=[None, config.bert_max_seq_length], name='bert_input_data_segment_ids') 
            self.bert_embeddings = self.__bert_embedding(self.bert_input_data_token_ids,
                                                         self.bert_input_data_token_masks,
                                                         self.bert_input_data_segment_ids,
                                                         masks,
                                                         keep_prob=bert_keep_prob)

        concat_in = [self.word_embeddings, self.wordchr_embeddings, self.pos_embeddings, self.chk_embeddings]
        if self.emb_class == 'elmo':
            concat_in = [self.word_embeddings, self.wordchr_embeddings, self.elmo_embeddings, self.pos_embeddings, self.chk_embeddings]
        if self.emb_class == 'bert':
            concat_in = [self.word_embeddings, self.wordchr_embeddings, self.bert_embeddings, self.pos_embeddings, self.chk_embeddings]
        if self.emb_class == 'bert+elmo':
            # we need to extend elmo_embeddings for bert(token based) via tf.gather_nd().
            self.bert_input_data_elmo_indices = tf.compat.v1.placeholder(tf.int32, shape=[None, None, 2], name='bert_input_data_elmo_indices') # (batch_size, bert_max_seq_length, 2)
            self.elmo_embeddings = tf.gather_nd(self.elmo_embeddings, self.bert_input_data_elmo_indices)
            concat_in = [self.word_embeddings, self.wordchr_embeddings, self.bert_embeddings, self.elmo_embeddings, self.pos_embeddings, self.chk_embeddings]
        self.input_data = tf.concat(concat_in, axis=-1, name='input_data') # (batch_size, sentence_length, input_dim)
        
        # highway network
        if config.highway_used:
            input_dim = self.input_data.get_shape()[-1]
            self.input_data = tf.reshape(self.input_data, [-1, input_dim]) 
            self.input_data = highway(self.input_data, input_dim, num_layers=2, scope='highway')
            self.input_data = tf.reshape(self.input_data, [-1, self.sentence_length, input_dim])
            self.input_data = tf.nn.dropout(self.input_data, rate=1 - (keep_prob))

        # masking (for confirmation)
        self.input_data *= masks

        """
        RNN layer
        """
        rnn_output = tf.identity(self.input_data)
        if config.rnn_used:
            for i in range(config.rnn_num_layers):
                if config.rnn_type == 'fused':
                    scope = 'bi-lstm-fused-%s' % i
                    rnn_output = self.__bi_lstm_fused(rnn_output,
                                                      self.sentence_lengths,
                                                      rnn_size=config.rnn_size,
                                                      keep_prob=keep_prob,
                                                      scope=scope) # (batch_size, sentence_length, 2*rnn_size)
                elif config.rnn_type == 'qrnn':
                    scope = 'bi-qrnn-%s' % i
                    xp = self.__projection(rnn_output,
                                           config.qrnn_size*2,
                                           scope='projection-%s' % scope) # (batch_size, sentence_length, config.qrnn_size*2)
                    x = xp
                    y = self.__bi_qrnn(xp,
                                       self.sentence_lengths,
                                       rnn_size=config.qrnn_size,
                                       keep_prob=1.0,
                                       scope=scope)   # (batch_size, sentence_length, input_dim)
                    # residual and dropout
                    rnn_output = tf.nn.dropout(y + x, rate=1 - (keep_prob))
                else:
                    scope = 'bi-lstm-%s' % i
                    rnn_output = self.__bi_lstm(rnn_output,
                                                self.sentence_lengths,
                                                rnn_size=config.rnn_size,
                                                keep_prob=keep_prob, scope=scope) # (batch_size, sentence_length, 2*rnn_size)
        self.rnn_output = rnn_output

        """
        Transformer layer
        """
        transformed_output = tf.identity(self.rnn_output)
        if config.tf_used:
            tf_keep_prob = tf.cond(pred=self.is_train, true_fn=lambda: config.tf_keep_prob, false_fn=lambda: 1.0)
            tf_mh_keep_prob = tf.cond(pred=self.is_train, true_fn=lambda: config.tf_mh_keep_prob, false_fn=lambda: 1.0)
            tf_ffn_keep_prob = tf.cond(pred=self.is_train, true_fn=lambda: config.tf_ffn_keep_prob, false_fn=lambda: 1.0)
            # last dimension must be equal to model_dim because we use a residual connection. 
            model_dim = transformed_output.get_shape().as_list()[-1]
            # sinusoidal positional signal
            signal = positional_encoding(self.sentence_lengths,
                                         self.sentence_length,
                                         model_dim,
                                         zero_pad=False,
                                         scale=False,
                                         scope='positional-encoding',
                                         reuse=None)
            transformed_output += signal
            # block
            for i in range(config.tf_num_layers):
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
                x = tf.nn.dropout(x_norm + y, rate=1 - (tf_keep_prob))
                # layer norm
                x_norm = normalize(x, scope='layer-norm-ffn-%s'%i, reuse=None)
                # position-wise feed forward net
                y = self.__feedforward(x_norm,
                                       masks,
                                       model_dim=model_dim,
                                       kernel_size=config.tf_ffn_kernel_size,
                                       keep_prob=tf_ffn_keep_prob,
                                       scope='feed-forward-%s'%i)
                # residual and dropout
                x = tf.nn.dropout(x_norm + y, rate=1 - (tf_keep_prob))
                transformed_output = x
            # final layer norm
            transformed_output = normalize(transformed_output, scope='layer-norm', reuse=None)
        self.transformed_output = transformed_output

        """
        Projection layer
        """
        self.logits = self.__projection(self.transformed_output,
                                        self.class_size,
                                        scope='projection') # (batch_size, sentence_length, class_size)

        """
        Output answer
        """
        self.output_data = tf.compat.v1.placeholder(tf.float32,
                                          shape=[None, None, self.class_size], # (batch_size, sentence_length, class_size)
                                          name='output_data')
        self.output_data_indices = tf.argmax(input=self.output_data, axis=-1, output_type=tf.int32) # (batch_size, sentence_length)

        """
        Loss, Prediction, Measures, Optimization
        """
        self.loss = self.__compute_loss()
        self.prediction = self.__compute_prediction()
        self.logits_indices = tf.identity(self.prediction, name='logits_indices')
        self.accuracy, self.precision, self.recall, self.f1 = self.__compute_measures()

        with tf.compat.v1.variable_scope('optimization'):
            self.global_step = tf.compat.v1.train.get_or_create_global_step()
            if 'bert' in config.emb_class:
                from bert import optimization
                if config.use_bert_optimization:
                    self.learning_rate = tf.constant(value=config.starter_learning_rate, shape=[], dtype=tf.float32)
                    self.train_op = optimization.create_optimizer(self.loss,
                                                                  config.starter_learning_rate,
                                                                  config.num_train_steps,
                                                                  config.num_warmup_steps,
                                                                  False)
                else:
                    # exponential decay of the learning rate
                    self.learning_rate = tf.compat.v1.train.exponential_decay(config.starter_learning_rate,
                                                                    self.global_step,
                                                                    config.decay_steps,
                                                                    config.decay_rate,
                                                                    staircase=True)
                    # linear warmup, if global_step < num_warmup_steps, then
                    # learning rate = (global_step / num_warmup_steps) * starter_learning_rate
                    global_steps_int = tf.cast(self.global_step, tf.int32)
                    warmup_steps_int = tf.constant(config.num_warmup_steps, dtype=tf.int32)
                    global_steps_float = tf.cast(global_steps_int, tf.float32)
                    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
                    warmup_percent_done = global_steps_float / warmup_steps_float
                    warmup_learning_rate = config.starter_learning_rate * warmup_percent_done
                    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
                    self.learning_rate = ((1.0 - is_warmup) * self.learning_rate + is_warmup * warmup_learning_rate)
                    # Adam optimizer with correct L2 weight decay
                    optimizer = optimization.AdamWeightDecayOptimizer(
                        learning_rate=self.learning_rate,
                        weight_decay_rate=0.01,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-6,
                        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
                    tvars = tf.compat.v1.trainable_variables()
                    grads, _ = tf.clip_by_global_norm(tf.gradients(ys=self.loss, xs=tvars), config.clip_norm)
                    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
                    new_global_step = self.global_step + 1
                    self.train_op = tf.group(train_op, [self.global_step.assign(new_global_step)])
            else:
                # exponential decay of the learning rate
                self.learning_rate = tf.compat.v1.train.exponential_decay(config.starter_learning_rate,
                                                                self.global_step,
                                                                config.decay_steps,
                                                                config.decay_rate,
                                                                staircase=True)
                # linear warmup, if global_step < num_warmup_steps, then
                # learning rate = (global_step / num_warmup_steps) * starter_learning_rate
                global_steps_int = tf.cast(self.global_step, tf.int32)
                warmup_steps_int = tf.constant(config.num_warmup_steps, dtype=tf.int32)
                global_steps_float = tf.cast(global_steps_int, tf.float32)
                warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
                warmup_percent_done = global_steps_float / warmup_steps_float
                warmup_learning_rate = config.starter_learning_rate * warmup_percent_done
                is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
                self.learning_rate = ((1.0 - is_warmup) * self.learning_rate + is_warmup * warmup_learning_rate)
                # Adam optimizer
                optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
                tvars = tf.compat.v1.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(ys=self.loss, xs=tvars), config.clip_norm)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
    
    def __word_embedding(self, inputs, keep_prob=0.5, scope='word-embedding'):
        """Look up word embeddings.
        """
        with tf.compat.v1.variable_scope(scope):
            with tf.device('/cpu:0'):
                word_embeddings = tf.nn.embedding_lookup(params=self.wrd_embeddings, ids=inputs) # (batch_size, sentence_length, wrd_dim)
            return tf.nn.dropout(word_embeddings, 1 - (keep_prob))

    def __wordchr_embedding_conv1d(self, inputs, keep_prob=0.5, scope='wordchr-embedding-conv1d'):
        """Compute character embeddings by masked conv1d and max-pooling.
        """
        with tf.compat.v1.variable_scope(scope):
            with tf.device('/cpu:0'):
                chr_embeddings = tf.Variable(tf.random.uniform([self.chr_vocab_size, self.chr_dim], -1.0, 1.0),
                                             name='chr_embeddings')
                wordchr_embeddings_t = tf.nn.embedding_lookup(params=chr_embeddings, ids=inputs) # (batch_size, sentence_length, word_length, chr_dim)
                wordchr_embeddings_t = tf.nn.dropout(wordchr_embeddings_t, 1 - (keep_prob))
            wordchr_embeddings_t = tf.reshape(wordchr_embeddings_t,
                                              [-1, self.word_length, self.chr_dim])   # (batch_size*sentence_length, word_length, chr_dim)
            # masking
            t = tf.reshape(inputs, [-1, self.word_length])  # (batch_size*sentence_length, word_length) 
            masks = self.__compute_word_masks(t)            # (batch_size*sentence_length, word_length)
            filters = self.config.num_filters
            kernel_size = self.config.filter_sizes[0]
            wordchr_embeddings = masked_conv1d_and_max(wordchr_embeddings_t, masks, filters, kernel_size, tf.nn.relu)
            # (batch_size*sentence_length, filters) -> (batch_size, sentence_length, filters)
            wordchr_embeddings = tf.reshape(wordchr_embeddings, [-1, self.sentence_length, filters])
            return tf.nn.dropout(wordchr_embeddings, 1 - (keep_prob))

    def __wordchr_embedding_conv2d(self, inputs, keep_prob=0.5, scope='wordchr-embedding-conv2d'):
        """Compute character embeddings by conv2d and max-pooling.
        """
        with tf.compat.v1.variable_scope(scope):
            with tf.device('/cpu:0'):
                chr_embeddings = tf.Variable(tf.random.uniform([self.chr_vocab_size, self.chr_dim], -1.0, 1.0),
                                              name='chr_embeddings')
                wordchr_embeddings_t = tf.nn.embedding_lookup(params=chr_embeddings, ids=inputs) # (batch_size, sentence_length, word_length, chr_dim)
            wordchr_embeddings_t = tf.reshape(wordchr_embeddings_t,
                                              [-1, self.word_length, self.chr_dim])   # (batch_size*sentence_length, word_length, chr_dim)
            # masking
            t = tf.reshape(inputs, [-1, self.word_length])   # (batch_size*sentence_length, word_length) 
            masks = self.__compute_word_masks(t)             # (batch_size*sentence_length, word_length)
            masks = tf.expand_dims(masks, -1)                # (batch_size*sentence_length, word_length, 1)
            wordchr_embeddings_t *= tf.cast(masks, dtype=tf.float32)       # broadcasting
            # conv and max-pooling
            wordchr_embeddings = tf.expand_dims(wordchr_embeddings_t, -1)   # (batch_size*sentence_length, word_length, chr_dim, 1)
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.compat.v1.variable_scope('conv-maxpool-%s' % filter_size):
                    # convolution layer
                    filter_shape = [filter_size, self.chr_dim, 1, self.config.num_filters]
                    W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name='W')
                    conv = tf.nn.conv2d(
                        input=wordchr_embeddings,
                        filters=W,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='conv')
                    # apply nonlinearity
                    b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name='b')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    # max-pooling over the outputs
                    pooled = tf.nn.max_pool2d(
                        input=h,
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
            num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
            h_pool = tf.concat(pooled_outputs, axis=-1)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            """
            h_pool Tensor("concat:0", shape=(?, 1, 1, num_filters_total), dtype=float32)
            h_pool_flat Tensor("Reshape:0", shape=(?, num_filters_total), dtype=float32)
            """
            # (batch_size*sentence_length, num_filters_total) -> (batch_size, sentence_length, num_filters_total)
            wordchr_embeddings = tf.reshape(h_pool_flat, [-1, self.sentence_length, num_filters_total])
            return tf.nn.dropout(wordchr_embeddings, 1 - (keep_prob))

    def __elmo_embedding(self, inputs, masks, keep_prob=0.8):
        """Compute ELMo embeddings.
        """
        from bilm import weight_layers
        elmo_embeddings_op = self.elmo_bilm(inputs)
        elmo_input = weight_layers('input', elmo_embeddings_op, l2_coef=0.0)
        elmo_embeddings = elmo_input['weighted_op'] # (batch_size, sentence_length, elmo_dim)
        # masking(remove noise due to padding)
        elmo_embeddings *= masks
        return tf.nn.dropout(elmo_embeddings, 1 - (keep_prob))

    def __bert_embedding(self, token_ids, token_masks, segment_ids, masks, keep_prob=0.8):
        """Compute BERT embeddings.
        """
        from bert import modeling
        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=token_ids,
            input_mask=token_masks,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)
        bert_embeddings = bert_model.get_sequence_output()  # (batch_size, bert_max_seq_length, bert_embedding_size)
        # initialize pre-trained bert
        if self.is_training and self.bert_init_checkpoint:
            tvars = tf.compat.v1.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, self.bert_init_checkpoint)
            tf.compat.v1.train.init_from_checkpoint(self.bert_init_checkpoint, assignment_map)
            tf.compat.v1.logging.debug("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.compat.v1.logging.debug("  name = %s, shape = %s%s", var.name, var.shape, init_string)
        return tf.nn.dropout(bert_embeddings, 1 - (keep_prob))

    def __pos_embedding(self, inputs, keep_prob=0.5, scope='pos-embedding'):
        """Computing pos embeddings.
        """
        with tf.compat.v1.variable_scope(scope):
            with tf.device('/cpu:0'):
                p_embeddings = tf.Variable(tf.random.uniform([self.pos_vocab_size, self.pos_dim], -0.5, 0.5),
                                           name='p_embeddings')
                pos_embeddings = tf.nn.embedding_lookup(params=p_embeddings, ids=inputs) # (batch_size, sentence_length, pos_dim)
            # masking
            masks = tf.expand_dims(self.sentence_masks, -1)                   # (batch_size, sentence_length, 1)
            pos_embeddings *= tf.cast(masks, dtype=tf.float32)                              # broadcasting
            return tf.nn.dropout(pos_embeddings, 1 - (keep_prob))

    def __chk_embedding(self, inputs, keep_prob=0.5, scope='chk-embedding'):
        """Computing chk embeddings.
        """
        with tf.compat.v1.variable_scope(scope):
            with tf.device('/cpu:0'):
                k_embeddings = tf.Variable(tf.random.uniform([self.chk_vocab_size, self.chk_dim], -0.5, 0.5),
                                           name='k_embeddings')
                chk_embeddings = tf.nn.embedding_lookup(params=k_embeddings, ids=inputs) # (batch_size, sentence_length, chk_dim)
            # masking
            masks = tf.expand_dims(self.sentence_masks, -1)                   # (batch_size, sentence_length, 1)
            chk_embeddings *= tf.cast(masks, dtype=tf.float32)                              # broadcasting
            return tf.nn.dropout(chk_embeddings, 1 - (keep_prob))

    def __bi_lstm(self, inputs, lengths, rnn_size, keep_prob=0.5, scope='bi-lstm'):
        """Apply bi-directional LSTM.
        """
        with tf.compat.v1.variable_scope(scope):
            cell_fw = tf.compat.v1.nn.rnn_cell.LSTMCell(rnn_size)
            cell_bw = tf.compat.v1.nn.rnn_cell.LSTMCell(rnn_size)
            (output_fw, output_bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                        cell_bw,
                                                                        inputs,
                                                                        sequence_length=lengths,
                                                                        dtype=tf.float32)
            outputs = tf.concat([output_fw, output_bw], axis=-1)
            return tf.nn.dropout(outputs, 1 - (keep_prob))

    def __bi_lstm_fused(self, inputs, lengths, rnn_size, keep_prob=0.5, scope='bi-lstm-fused'):
        """Apply bi-directional LSTM block fused.
        """
        with tf.compat.v1.variable_scope(scope):
            t = tf.transpose(a=inputs, perm=[1, 0, 2])  # Need time-major
            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(rnn_size)
            lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(rnn_size)
            lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
            output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=lengths)
            output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=lengths)
            outputs = tf.concat([output_fw, output_bw], axis=-1)
            outputs = tf.transpose(a=outputs, perm=[1, 0, 2])
            return tf.nn.dropout(outputs, 1 - (keep_prob))

    def __bi_qrnn(self, inputs, lengths, rnn_size, keep_prob=0.5, scope='bi-qrnn'):
        """Apply bi-directional Quasi-RNN
        """
        import qrnn
        with tf.compat.v1.variable_scope(scope):
            # forward
            inputs_fw = inputs
            outputs_fw, _ = qrnn.qrnn(inputs_fw, num_outputs=rnn_size, window=self.config.qrnn_filter_size, scope=scope+'-fw')
            # backward
            inputs_bw = tf.reverse_sequence(input=inputs, seq_lengths=lengths, batch_axis=0, seq_axis=1)
            outputs_bw, _ = qrnn.qrnn(inputs_bw, num_outputs=rnn_size, window=self.config.qrnn_filter_size, scope=scope+'-bw')
            outputs_bw = tf.reverse_sequence(input=outputs_bw, seq_lengths=lengths, batch_axis=0, seq_axis=1)
            outputs = tf.concat([outputs_fw, outputs_bw], axis=-1)
            return tf.nn.dropout(outputs, 1 - (keep_prob))

    def __self_attention(self, inputs, masks, model_dim=None, keep_prob=0.5, scope='self-attention'):
        """Apply self attention.
        """
        with tf.compat.v1.variable_scope(scope):
            inputs *= masks # inputs should be masked before multihead_attention()
            if not model_dim: model_dim = inputs.get_shape().as_list()[-1]
            queries = inputs
            keys = inputs
            attended_queries = multihead_attention(queries,
                                                   keys,
                                                   num_units=self.config.tf_mh_num_units,
                                                   num_heads=self.config.tf_mh_num_heads,
                                                   model_dim=model_dim,
                                                   dropout_rate=1.0 - keep_prob,
                                                   is_training=self.is_training,
                                                   causality=False, # no future masking
                                                   scope='multihead-attention',
                                                   reuse=None)
            return attended_queries

    def __feedforward(self, inputs, masks, model_dim=None, kernel_size=1, keep_prob=0.5, scope='feed-forward'):
        """Apply Point-wise feed forward layer.
        """
        with tf.compat.v1.variable_scope(scope):
            if not model_dim: model_dim = inputs.get_shape().as_list()[-1]
            num_units = [4*model_dim, model_dim]
            outputs = feedforward(inputs, masks, num_units=num_units, kernel_size=kernel_size, scope=scope, reuse=None)
            outputs = tf.nn.dropout(outputs, 1 - (keep_prob))
            return outputs

    def __projection(self, inputs, out_dim, scope='projection'):
        """Apply fully-connected projection layer.
        """
        with tf.compat.v1.variable_scope(scope):
            in_dim = inputs.get_shape().as_list()[-1]
            weight = tf.compat.v1.get_variable('W', shape=[in_dim, out_dim],
                                     dtype=tf.float32, initializer=tf.compat.v1.truncated_normal_initializer(), use_resource=False)
            bias = tf.compat.v1.get_variable('b', shape=[out_dim], dtype=tf.float32,
                                   initializer=tf.compat.v1.initializers.zeros(), use_resource=False)
            t_output = tf.reshape(inputs, [-1, in_dim])                      # (batch_size*sentence_length, in_dim)
            output = tf.matmul(t_output, weight) + bias                      # (batch_size*sentence_length, out_dim)
            output = tf.reshape(output, [-1, self.sentence_length, out_dim]) # (batch_size, sentence_length, out_dim)
            return output

    def __compute_loss(self):
        """Compute loss(self.output_data, self.logits).
        """
        trans_params = tf.compat.v1.get_variable('trans_params',
                                       shape=[self.class_size, self.class_size],
                                       initializer=tf.compat.v1.truncated_normal_initializer(), use_resource=False)
        self.trans_params = trans_params
        if self.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(inputs=self.logits,
                                                                             tag_indices=self.output_data_indices,
                                                                             transition_params=trans_params,
                                                                             sequence_lengths=self.sentence_lengths)
            return tf.reduce_mean(input_tensor=-log_likelihood)
        else:
            cross_entropy = self.output_data * tf.math.log(tf.nn.softmax(self.logits)) # (batch_size, sentence_length, class_size)
            cross_entropy = -tf.reduce_sum(input_tensor=cross_entropy, axis=2)    # (batch_size, sentence_length)
            # masking
            masks = self.sentence_masks 
            cross_entropy *= tf.cast(masks, dtype=tf.float32)
            cross_entropy = tf.reduce_sum(input_tensor=cross_entropy, axis=1)     # (batch_size)
            cross_entropy /= tf.cast(self.sentence_lengths, tf.float32)           # (batch_size)
            trans_params = tf.constant(0.0, shape=[self.class_size, self.class_size])
            return tf.reduce_mean(input_tensor=cross_entropy)

    def __compute_prediction(self):
        """Compute prediction(self.logits, self.trans_params).
        """
        if self.use_crf:
            prediction, _ = tf.contrib.crf.crf_decode(potentials=self.logits,
                                                      transition_params=self.trans_params,
                                                      sequence_length=self.sentence_lengths)
        else:
            probabilities = tf.nn.softmax(self.logits, axis=-1)                    # (batch_size, sentence_length, class_size)
            prediction = tf.argmax(input=probabilities, axis=-1, output_type=tf.int32)   # (batch_size, sentence_length)
        return prediction

    def __compute_measures(self):
        """Compute measures(self.prediction, self.output_data_indices).
        """
        masks = self.sentence_masks

        # compute accuracy
        correct_prediction = tf.cast(tf.equal(self.prediction, self.output_data_indices),
                                     tf.float32)                                     # (batch_size, sentence_length)
        correct_prediction *= tf.cast(masks, dtype=tf.float32)
        correct_prediction = tf.reduce_sum(input_tensor=correct_prediction, axis=1)  # (batch_size)
        correct_prediction /= tf.cast(self.sentence_lengths, tf.float32)             # (batch_size)
        accuracy = tf.reduce_mean(input_tensor=correct_prediction)

        # compute precision, recall, f1
        indices = [i for i in range(2, self.class_size)] # ignore '0' for 'O', '1' for 'X'
        prec, prec_op = tf_metrics.precision(self.output_data_indices, self.prediction, self.class_size, indices, masks)
        rec, rec_op = tf_metrics.recall(self.output_data_indices, self.prediction, self.class_size, indices, masks)
        f1, f1_op = tf_metrics.f1(self.output_data_indices, self.prediction, self.class_size, indices, masks)
        return accuracy, prec_op, rec_op, f1_op

    def __compute_sentence_lengths(self, sentence_masks):
        """Compute each sentence lengths.
        """
        return tf.cast(tf.reduce_sum(input_tensor=sentence_masks, axis=1), tf.int32) # (batch_size)

    def __compute_sentence_masks(self, t):
        """Compute each sentence masks.
        """
        sentence_masks = tf.sign(tf.abs(t)) # (batch_size, sentence_length)
        return sentence_masks

    def __compute_word_masks(self, t):
        """Compute each word masks.
        """
        word_masks = tf.sign(tf.abs(t))    # (batch_size*sentence_length, word_length)
        return word_masks

    @staticmethod
    def set_cuda_visible_devices(is_training):
        import os
        os.environ['CUDA_VISIBLE_DEVICES']='1'
        if is_training:
            from tensorflow.python.client import device_lib
            print(device_lib.list_local_devices())
        return True
