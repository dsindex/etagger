from __future__ import print_function
import numpy as np
import pickle as pkl

class Config:
    def __init__(self, args, arg_train=True, emb_class='glove', use_crf=True):
        self.emb_path = args.emb_path
        self.embvec = pkl.load(open(self.emb_path, 'rb')) # resources(glove, vocab, path, etc)
        self.wrd_dim = args.wrd_dim         # size of word embedding(glove)
        self.chr_dim = 50                   # size of character embedding
        self.pos_dim = 7                    # size of part of speech embedding
        self.class_size = len(self.embvec.tag_vocab) # number of class(tags)
        self.word_length = args.word_length # maximum character size of word for convolution
        self.restore = args.restore         # checkpoint path if available
        self.use_crf = use_crf              # use crf decoder or not
        self.emb_class = emb_class          # class of embedding(glove, elmo, bert)
        self.starter_learning_rate = 0.001  # 0.001(default), 0.0003(transformer)
        self.decay_steps = 12000
        self.decay_rate = 0.9
        self.clip_norm = 10

        self.keep_prob = 0.7                # keep probability for dropout
        self.chr_conv_type = 'conv1d'       # conv1d | conv2d
        self.filter_sizes = [3]             # filter sizes
        self.num_filters = 25               # number of filters
        self.rnn_used = True                # use rnn layer or not
        self.rnn_num_layers = 2             # number of RNN layers
        self.rnn_type = 'fused'             # normal | fused
        self.rnn_size = 200                 # size of RNN hidden unit
        self.tf_used = False                # use transformer encoder layer or not
        self.tf_num_layers = 4              # number of layers for transformer encoder
        self.tf_keep_prob = 0.8             # keep probability for transformer encoder
        self.tf_mh_num_heads = 4            # number of head for multi head attention
        self.tf_mh_num_units = 64           # Q,K,V dimension for multi head attention
        self.tf_mh_keep_prob = 0.8          # keep probability for multi head attention
        self.tf_ffn_kernel_size = 3         # conv1d kernel size for feed forward net
        self.tf_ffn_keep_prob = 0.8         # keep probability for feed forward net

        self.arg_train = arg_train
        if self.arg_train:
            self.epoch = args.epoch
            self.batch_size = args.batch_size
            self.dev_batch_size = 2*self.batch_size
            self.checkpoint_dir = args.checkpoint_dir
            self.summary_dir = args.summary_dir

        if self.emb_class == 'elmo':
            from bilm import Batcher, BidirectionalLanguageModel
            self.word_length = 50 # replace to fixed word length for the pre-trained elmo : 'max_characters_per_token'
            self.elmo_batcher = Batcher(self.embvec.elmo_vocab_path, self.word_length) # map text to character ids
            self.elmo_bilm = BidirectionalLanguageModel(self.embvec.elmo_options_path, self.embvec.elmo_weight_path) # biLM graph
            self.elmo_keep_prob = 0.8
        if self.emb_class == 'bert':
            from bert import modeling
            from bert import tokenization
            self.bert_config = modeling.BertConfig.from_json_file(self.embvec.bert_config_path)
            self.bert_tokenizer = tokenization.FullTokenizer(
                vocab_file=self.embvec.bert_vocab_path, do_lower_case=self.embvec.bert_do_lower_case)
            self.bert_init_checkpoint = self.embvec.bert_init_checkpoint
            self.bert_max_seq_length = self.embvec.bert_max_seq_length
            self.bert_keep_prob = 0.8
            # modified for bert
            self.rnn_size = 256
            self.starter_learning_rate = 2e-5
            self.decay_steps = 5000
            self.decay_rate = 0.9
            self.clip_norm = 1.5
            if self.arg_train:
                self.dev_batch_size = self.batch_size # set batch_size == dev_batch_size
