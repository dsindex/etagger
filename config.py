from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle as pkl

'''
etc dimension
  : you should define etc dimension by refering __create_etc() of input.txt
'''
ETC_DIM = 5+5+1

'''
character dimension
'''
CHR_DIM = 96

class Config:
    def __init__(self, args, is_train=1):
        self.emb_path = args.emb_path
        self.embvec = pkl.load(open(self.emb_path, 'rb'))
        self.wrd_dim = args.wrd_dim
        self.etc_dim = ETC_DIM
        self.chr_dim = CHR_DIM
        self.class_size = len(self.embvec.tag_vocab)
        self.sentence_length = args.sentence_length
        self.word_length = args.word_length
        self.restore = args.restore
        if is_train:
            self.epoch = args.epoch
            self.batch_size = args.batch_size
            self.checkpoint_dir = args.checkpoint_dir
        self.is_train = is_train
