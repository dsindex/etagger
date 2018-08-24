from __future__ import print_function
import tensorflow as tf
import numpy as np
from embvec import EmbVec
import sys

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
            etc, tag = self.__create_etc(bucket)
            self.sentence_etc.append(etc)
            self.sentence_tag.append(tag)
        else:                  # treat data as file path
            path = data
            bucket = []
            for line in open(path):
                if line in ['\n', '\r\n']:
                    word_ids = self.__create_word_ids(bucket)
                    self.sentence_word_ids.append(word_ids)
                    wordchr_ids = self.__create_wordchr_ids(bucket)
                    self.sentence_wordchr_ids.append(wordchr_ids)
                    etc, tag = self.__create_etc(bucket)
                    self.sentence_etc.append(etc)
                    self.sentence_tag.append(tag)
                    bucket = []
                else:
                    bucket.append(line)

    def __create_word_ids(self, bucket):
        word_ids = []
        sentence_length = 0
        for line in bucket:
            tokens = line.split()
            assert (len(tokens) == 4)
            sentence_length += 1
            wid = self.config.embvec.get_wid(tokens[0])
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
            tokens = line.split()
            assert (len(tokens) == 4)
            sentence_length += 1
            chr_ids = []
            word_length = 0
            for ch in tokens[0]:
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

    def __create_etc(self, bucket):
        etc = []
        tag  = []
        sentence_length = 0
        for line in bucket:
            tokens = line.split()
            assert (len(tokens) == 4)
            sentence_length += 1
            temp = self.pos(tokens[1])                                # adding pos one-hot(5)
            temp = np.append(temp, self.chunk(tokens[2]))             # adding chunk one-hot(5)
            temp = np.append(temp, self.capital(tokens[0]))           # adding capital one-hot(1)
            etc.append(temp)
            tag.append(self.label(tokens[3], self.config.class_size)) # label one-hot(9)
            if sentence_length == self.max_sentence_length: break
        # padding with 0s
        for _ in range(self.max_sentence_length - sentence_length):
            temp = np.array([0 for _ in range(self.config.etc_dim)])
            etc.append(temp)
            tag.append(np.array([0] * self.config.class_size))
        return etc, tag

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
    def pos(tag):
        one_hot = np.zeros(5)
        if tag == 'NN' or tag == 'NNS':
            one_hot[0] = 1
        elif tag == 'FW':
            one_hot[1] = 1
        elif tag == 'NNP' or tag == 'NNPS':
            one_hot[2] = 1
        elif 'VB' in tag:
            one_hot[3] = 1
        else:
            one_hot[4] = 1
        return one_hot

    @staticmethod
    def chunk(tag):
        one_hot = np.zeros(5)
        if 'NP' in tag:
            one_hot[0] = 1
        elif 'VP' in tag:
            one_hot[1] = 1
        elif 'PP' in tag:
            one_hot[2] = 1
        elif tag == 'O':
            one_hot[3] = 1
        else:
            one_hot[4] = 1
        return one_hot

    @staticmethod
    def capital(word):
        one_hot = np.zeros(1)
        if ord('A') <= ord(word[0]) <= ord('Z'):
            one_hot[0] = 1
        return one_hot

    @staticmethod
    def label(tag, class_size):
        one_hot = np.zeros(class_size)
        if tag == 'B-PER':
            one_hot[0] = 1
        elif tag == 'I-PER':
            one_hot[1] = 1
        elif tag == 'B-LOC':
            one_hot[2] = 1
        elif tag == 'I-LOC':
            one_hot[3] = 1
        elif tag == 'B-ORG':
            one_hot[4] = 1
        elif tag == 'I-ORG':
            one_hot[5] = 1
        elif tag == 'B-MISC':
            one_hot[6] = 1
        elif tag == 'I-MISC':
            one_hot[7] = 1
        else:
            one_hot[8] = 1
        return one_hot

    @staticmethod
    def pred_to_label(pred, length):
        '''
        pred : [senence_length, class_size]
        length : int
        '''
        pred = pred[0:length]
        # [length]
        pred_list = np.argmax(pred, 1).tolist()
        labels = []
        for i in pred_list:
            if i == 0: labels.append('B-PER')
            elif i == 1: labels.append('I-PER')
            elif i == 2: labels.append('B-LOC')
            elif i == 3: labels.append('I-LOC')
            elif i == 4: labels.append('B-ORG')
            elif i == 5: labels.append('I-ORG')
            elif i == 6: labels.append('B-MISC')
            elif i == 7: labels.append('I-MISC')
            else: labels.append('O')
        return labels

