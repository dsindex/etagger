from __future__ import print_function
import tensorflow as tf
import numpy as np
from embvec import EmbVec
from shuffle import Shuffle
import sys
import re

class Input:
    def __init__(self, data, config):
        self.sentence_word_ids = []      # [None, sentence_length]
        self.sentence_wordchr_ids = []   # [None, sentence_length, word_length]
        self.sentence_etc = []           # [None, sentence_length, etc_dim]
        self.sentence_tag = []
        self.config = config
        if config.sentence_length == -1:
            if type(data) is list:
                self.max_sentence_length = len(data)
            else: # treat as file path
                self.max_sentence_length = self.find_max_length(data)
        else:
            self.max_sentence_length = config.sentence_length

        if type(data) is list: # treat data as bucket
            bucket = data
            word_ids = self.__create_word_ids(bucket)
            self.sentence_word_ids.append(word_ids)
            wordchr_ids = self.__create_wordchr_ids(bucket)
            self.sentence_wordchr_ids.append(wordchr_ids)
            etc, tag = self.__create_etc_and_tag(bucket)
            self.sentence_etc.append(etc)
            self.sentence_tag.append(tag)
        else:                  # treat data as file path
            path = data
            '''
            if config.is_train:
                sp = Shuffle()
                sp.add(path)
                path = path + '.shuffle'
                sp.shuffle(path)
            '''
            bucket = []
            for line in open(path):
                if line in ['\n', '\r\n']:
                    word_ids = self.__create_word_ids(bucket)
                    self.sentence_word_ids.append(word_ids)
                    wordchr_ids = self.__create_wordchr_ids(bucket)
                    self.sentence_wordchr_ids.append(wordchr_ids)
                    etc, tag = self.__create_etc_and_tag(bucket)
                    self.sentence_etc.append(etc)
                    self.sentence_tag.append(tag)
                    bucket = []
                else:
                    bucket.append(line)

    def __create_word_ids(self, bucket):
        word_ids = []
        sentence_length = 0
        for line in bucket:
            line = line.strip()
            tokens = line.split()
            assert (len(tokens) == 4)
            sentence_length += 1
            word = self.replace_digits(tokens[0])
            wid = self.config.embvec.get_wid(word)
            word_ids.append(wid)
            if sentence_length == self.max_sentence_length: break
        # padding with pad wid
        for _ in range(self.max_sentence_length - sentence_length):
            word_ids.append(self.config.embvec.pad_wid)
        return word_ids

    def __create_wordchr_ids(self, bucket):
        wordchr_ids = []
        sentence_length = 0
        for line in bucket:
            line = line.strip()
            tokens = line.split()
            assert (len(tokens) == 4)
            sentence_length += 1
            chr_ids = []
            word_length = 0
            word = self.replace_digits(tokens[0])
            for ch in word:
                word_length += 1 
                cid = self.config.embvec.get_cid(ch)
                chr_ids.append(cid)
                if word_length == self.config.word_length: break
            # padding with pad cid
            for _ in range(self.config.word_length - word_length):
                chr_ids.append(self.config.embvec.pad_cid)
            wordchr_ids.append(chr_ids)
            if sentence_length == self.max_sentence_length: break
        # padding with empty chr_ids
        for _ in range(self.max_sentence_length - sentence_length):
            chr_ids = []
            for _ in range(self.config.word_length):
                chr_ids.append(self.config.embvec.pad_cid)
            wordchr_ids.append(chr_ids)
        return wordchr_ids

    def __create_etc_and_tag(self, bucket):
        etc = []
        tag  = []
        nbucket = []
        # apply gazetteer feature
        for line in bucket:
            line = line.strip()
            tokens = line.split()
            assert (len(tokens) == 4)
            gvec = np.zeros(self.config.class_size)
            tokens.append(gvec)
            nbucket.append(tokens)
        bucket_size = len(nbucket)
        i = 0
        while 1:
            if i >= bucket_size: break
            tokens = nbucket[i]
            j = self.config.embvec.apply_gaz(nbucket, bucket_size, i)
            i += j # jump
            i += 1
        sentence_length = 0
        for tokens in nbucket:
            sentence_length += 1
            word = self.replace_digits(tokens[0]).lower()
            temp = self.pos_vec(tokens[1])                                # adding pos one-hot(5)
            temp = np.append(temp, self.chunk_vec(tokens[2]))             # adding chunk one-hot(5)
            temp = np.append(temp, self.capital_vec(tokens[0]))           # adding capital one-hot(1)
            '''
            temp = np.append(temp, tokens[4])                             # adding gazetteer feature
            '''
            etc.append(temp)
            tag.append(self.tag_vec(tokens[3], self.config.class_size))   # tag one-hot(9)
            if sentence_length == self.max_sentence_length: break
        # padding with 0s
        for _ in range(self.max_sentence_length - sentence_length):
            temp = np.array([0 for _ in range(self.config.etc_dim)])
            etc.append(temp)
            tag.append(np.array([0] * self.config.class_size))
        return etc, tag

    def pos_vec(self, t):
        # language specific features
        one_hot = np.zeros(5)
        if t == 'NN' or t == 'NNS':
            one_hot[0] = 1
        elif t == 'FW':
            one_hot[1] = 1
        elif t == 'NNP' or t == 'NNPS':
            one_hot[2] = 1
        elif 'VB' in t:
            one_hot[3] = 1
        else:
            one_hot[4] = 1
        return one_hot

    def chunk_vec(self, t):
        # language specific features
        one_hot = np.zeros(5)
        if 'NP' in t:
            one_hot[0] = 1
        elif 'VP' in t:
            one_hot[1] = 1
        elif 'PP' in t:
            one_hot[2] = 1
        elif t == 'O':
            one_hot[3] = 1
        else:
            one_hot[4] = 1
        return one_hot

    def capital_vec(self, word):
        # language specific features
        one_hot = np.zeros(1)
        if ord('A') <= ord(word[0]) <= ord('Z'):
            one_hot[0] = 1
        return one_hot

    def tag_vec(self, tag, class_size):
        one_hot = np.zeros(class_size)
        tid = self.config.embvec.get_tid(tag)
        one_hot[tid] = 1
        return one_hot

    def pred_to_tags(self, pred, length):
        '''
        pred : [senence_length, class_size]
        length : int
        '''
        pred = pred[0:length]
        # [length]
        pred_list = np.argmax(pred, 1).tolist()
        tags = []
        for tid in pred_list:
            tag = self.config.embvec.get_tag(tid)
            tags.append(tag)
        return tags

    @staticmethod
    def find_max_length(file_name):
        temp_len = 0
        max_length = 0
        for line in open(file_name):
            if line in ['\n', '\r\n']:
                if temp_len > max_length:
                    max_length = temp_len
                temp_len = 0
            else:
                temp_len += 1
        return max_length

    @staticmethod
    def replace_digits(string):
        return string
        '''
        return re.sub('[0-9]', '0', string)
        '''

