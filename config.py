from __future__ import print_function
import sys
import numpy as np
import pickle as pkl
import json

class Config:

    def __init__(self, args, is_training=True, emb_class='glove', use_crf=True):
        """Set all parameters for model.

        Args:
          args: args from train.py, inference,py.
          is_training: True for training(from train.py), False for inference(inference.py)
          emb_class: class of embedding, glove | elmo | bert | bert+elmo.
          use_crf: if True, use crf decoder(bypass).
        """
        config = self.__load_config(args)
        self.emb_path = args.emb_path
        self.embvec = pkl.load(open(self.emb_path, 'rb'))             # resources(glove, vocab, path, etc)
        self.wrd_dim = args.wrd_dim                                   # dim of word embedding(glove)
        self.chr_dim = config['chr_dim']                              # dim of character embedding
        self.pos_dim = config['pos_dim']                              # dim of part of speech embedding
        self.chk_dim = config['chk_dim']                              # dim of chunk embedding
        self.class_size = len(self.embvec.tag_vocab)                  # number of class(tags)
        self.word_length = args.word_length                           # maximum character size of word for convolution
        self.restore = args.restore                                   # checkpoint path if available
        self.use_crf = use_crf                                        # use crf decoder or not
        self.emb_class = emb_class                                    # class of embedding(glove, elmo, bert, bert+elmo)

        self.keep_prob = config['keep_prob']                          # keep probability for dropout
        self.chr_conv_type = config['chr_conv_type']                  # conv1d | conv2d
        self.filter_sizes = config['filter_sizes']                    # filter sizes
        self.num_filters = config['num_filters']                      # number of filters
        self.highway_used = config['highway_used']                    # use highway network on the concatenated input
        self.rnn_used = config['rnn_used']                            # use rnn layer or not
        self.rnn_num_layers = config['rnn_num_layers']                # number of RNN layers
        self.rnn_type = config['rnn_type']                            # normal | fused | qrnn
        self.rnn_size = config['rnn_size']                            # size of RNN hidden unit
        self.tf_used = config['tf_used']                              # use transformer encoder layer or not
        self.tf_num_layers = config['tf_num_layers']                  # number of layers for transformer encoder
        self.tf_keep_prob = config['tf_keep_prob']                    # keep probability for transformer encoder
        self.tf_mh_num_heads = config['tf_mh_num_heads']              # number of head for multi head attention
        self.tf_mh_num_units = config['tf_mh_num_units']              # Q,K,V dimension for multi head attention
        self.tf_mh_keep_prob = config['tf_mh_keep_prob']              # keep probability for multi head attention
        self.tf_ffn_kernel_size = config['tf_ffn_kernel_size']        # conv1d kernel size for feed forward net
        self.tf_ffn_keep_prob = config['tf_ffn_keep_prob']            # keep probability for feed forward net

        self.starter_learning_rate = config['starter_learning_rate']  # default learning rate
        self.num_train_steps = 0                                      # number of total training steps, assigned by update()
        self.num_warmup_epoch = config['num_warmup_epoch']            # number of warmup epoch
        self.num_warmup_steps = 0                                     # number of warmup steps, assigned by update()
        self.decay_steps = config['decay_steps']
        self.decay_rate = config['decay_rate']
        self.clip_norm = config['clip_norm']
        if self.tf_used:                                              # modified for transformer
            self.starter_learning_rate = config['starter_learning_rate_for_tf']
        if self.rnn_type == 'qrnn':                                   # modified for QRNN
            self.qrnn_size = config['qrnn_size']                      # size of QRNN hidden units(number of filters)
            self.qrnn_filter_size = config['qrnn_filter_size']        # size of filter for QRNN
            self.rnn_num_layers = config['qrnn_num_layers']

        self.is_training = is_training
        if self.is_training:
            self.epoch = args.epoch
            self.batch_size = args.batch_size
            self.checkpoint_dir = args.checkpoint_dir
            self.summary_dir = args.summary_dir

        '''for CRZ wighout chk
        self.chk_dim = 10
        self.highway_used = False
        '''
        '''for CRZ with chk
        self.chk_dim = 64
        self.highway_used = True
        '''
        
        if 'elmo' in self.emb_class:
            from bilm import Batcher, BidirectionalLanguageModel
            self.word_length = config['elmo_word_length'] # replace to fixed word length for the pre-trained elmo : 'max_characters_per_token'
            self.elmo_batcher = Batcher(self.embvec.elmo_vocab_path, self.word_length) # map text to character ids
            self.elmo_bilm = BidirectionalLanguageModel(self.embvec.elmo_options_path, self.embvec.elmo_weight_path) # biLM graph
            self.elmo_keep_prob = config['elmo_keep_prob']
            '''for KOR
            self.rnn_size = 250
            '''
        if 'bert' in self.emb_class:
            from bert import modeling
            from bert import tokenization
            self.bert_config = modeling.BertConfig.from_json_file(self.embvec.bert_config_path)
            self.bert_tokenizer = tokenization.FullTokenizer(
                vocab_file=self.embvec.bert_vocab_path, do_lower_case=self.embvec.bert_do_lower_case)
            self.bert_init_checkpoint = self.embvec.bert_init_checkpoint
            self.bert_max_seq_length = self.embvec.bert_max_seq_length
            self.bert_dim = self.embvec.bert_dim
            self.bert_keep_prob = config['bert_keep_prob']
            self.use_bert_optimization = config['use_bert_optimization']
            self.num_warmup_epoch = config['num_warmup_epoch_for_bert']
            '''for KOR, CRZ
            self.rnn_size = 256
            self.starter_learning_rate = 5e-5
            self.num_warmup_epoch = 1
            self.decay_steps = 5000
            '''
            '''for KOR(CLOVA NER)
            self.pos_dim = 100
            self.starter_learning_rate = 5e-5
            self.num_warmup_epoch = 3
            self.decay_rate = 1.0
            '''

    def __load_config(self, args):
        """Load config from file.
        """
        try:
            with open(args.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            config = dict()
        return config

    def update(self, data):
        """Update num_train_steps, num_warmup_steps after reading training data

        Args:
          data: an instance of Input class, training data.
        """
        if not self.is_training: return False
        self.num_train_steps = int((data.num_examples / self.batch_size) * self.epoch)
        self.num_warmup_steps = self.num_warmup_epoch * int(data.num_examples / self.batch_size)
        if self.num_warmup_steps == 0: self.num_warmup_steps = 1 # prevent dividing by zero
        return True

# -----------------------------------------------------------------------------
# utility
# -----------------------------------------------------------------------------
            
    def logit_to_tags(self, logit, length):
        """Convert logit to tags.

        Args:
          logit: [sentence_length, class_size]
          length: int
        Returns:
          tag sequence(size length)
        """
        logit = logit[0:length]
        # [length]
        pred_list = np.argmax(logit, 1).tolist()
        tags = []
        for tid in pred_list:
            tag = self.embvec.get_tag(tid)
            tags.append(tag)
        return tags

    def logit_indices_to_tags(self, logit_indices, length):
        """Convert logit_indices to tags.

        Args:
          logit_indices: [sentence_length]
          length: int
        Returns:
          tag sequence(size length)
        """
        pred_list = logit_indices[0:length]
        tags = []
        for tid in pred_list:
            tag = self.embvec.get_tag(tid)
            tags.append(tag)
        return tags

    def logits_indices_to_tags_seq(self, logits_indices, lengths):
        """Convert logits_indices to sequence of tags.

        Args:
          logits_indices: [batch_size, sentence_length]
          lengths: [batch_size]
        Returns:
          sequence of tags
        """
        tags_seq = []
        for logit_indices, length in zip(logits_indices, lengths):
            tags = self.logit_indices_to_tags(logit_indices, length)
            tags_seq.append(tags)
        return tags_seq
