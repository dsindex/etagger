from __future__ import print_function
import numpy as np
import pickle as pkl

"""
etc dimension
  you should define etc dimension by refering __create_etc_and_tag() of input.txt
  shape vec(9) + pos one-hot(5) + [optional] chunk one-hot(5)
"""
ETC_DIM = 9 + 5

class Config:
    def __init__(self, args, is_train=True, use_crf=True):
        self.emb_path = args.emb_path
        self.embvec = pkl.load(open(self.emb_path, 'rb'))
        self.wrd_dim = args.wrd_dim
        self.chr_dim = 30
        self.pos_dim = 5
        # basic features + gazetteer feature
        '''
        self.etc_dim = ETC_DIM + len(self.embvec.tag_vocab)
        '''
        self.etc_dim = ETC_DIM
        self.class_size = len(self.embvec.tag_vocab)
        self.sentence_length = args.sentence_length
        self.word_length = args.word_length
        self.restore = args.restore
        self.use_crf = use_crf
        self.starter_learning_rate = 0.001
        self.decay_steps = 15000 # batch_size(20), epoch(20)
        self.decay_rate = 0.6
        self.is_train = is_train
        if self.is_train:
            self.epoch = args.epoch
            self.batch_size = args.batch_size
            self.checkpoint_dir = args.checkpoint_dir
            self.summary_dir = args.summary_dir
