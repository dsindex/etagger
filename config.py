from __future__ import print_function
import numpy as np
import pickle as pkl

class Config:
    def __init__(self, args, arg_train=True, emb_class='glove', use_crf=True):
        self.emb_path = args.emb_path
        self.embvec = pkl.load(open(self.emb_path, 'rb'))
        self.wrd_dim = args.wrd_dim
        self.chr_dim = 50
        self.pos_dim = 7
        self.class_size = len(self.embvec.tag_vocab)
        self.word_length = args.word_length
        self.restore = args.restore
        self.use_crf = use_crf
        self.emb_class = emb_class
        if self.emb_class == 'elmo':
            from bilm import Batcher, BidirectionalLanguageModel
            self.pos_dim = 8
            self.word_length = 50 # replace to fixed word length for the pre-trained elmo : 'max_characters_per_token'
            self.elmo_batcher = Batcher(self.embvec.elmo_vocab_path, self.word_length) # map text to character ids
            self.elmo_bilm = BidirectionalLanguageModel(self.embvec.elmo_options_path, self.embvec.elmo_weight_path) # biLM graph
        self.starter_learning_rate = 0.001 # 0.0003
        self.decay_steps = 12000
        self.decay_rate = 0.9
        if arg_train:
            self.epoch = args.epoch
            self.batch_size = args.batch_size
            self.dev_batch_size = 2*self.batch_size
            self.checkpoint_dir = args.checkpoint_dir
            self.summary_dir = args.summary_dir
        self.arg_train = arg_train
