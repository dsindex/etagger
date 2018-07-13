from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle as pkl

# see input.py -> __create_etc()
# you should define etc dimension.
ETC_DIM = 5+5+1

class Config:
    def __init__(self, args, is_train=1):
        self.emb_path = args.emb_path
        self.embvec = pkl.load(open(self.emb_path, 'rb'))
        self.emb_dim = args.emb_dim
        self.etc_dim = ETC_DIM
        self.class_size = args.class_size
        self.sentence_length = args.sentence_length
        self.restore = args.restore
        if is_train:
            self.epoch = args.epoch
            self.batch_size = args.batch_size
            self.checkpoint_dir = args.checkpoint_dir
        self.is_train = is_train
