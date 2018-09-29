from __future__ import print_function
import sys
import re
import tensorflow as tf
import numpy as np
from embvec import EmbVec

class Input:
    def __init__(self, data, config):
        self.sentence_word_ids = []      # [batch_size, sentence_length]
        self.sentence_wordchr_ids = []   # [batch_size, sentence_length, word_length]
        self.sentence_pos_ids = []       # [batch_size, sentence_length]
        self.sentence_etcs = []          # [batch_size, sentence_length, etc_dim]
        self.sentence_tags = []          # [batch_size, sentence_length, class_size] 
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
            pos_ids = self.__create_pos_ids(bucket)
            self.sentence_pos_ids.append(pos_ids)
            etc, tag = self.__create_etc_and_tag(bucket)
            self.sentence_etcs.append(etc)
            self.sentence_tags.append(tag)
        else:                  # treat data as file path
            path = data
            bucket = []
            for line in open(path):
                if line in ['\n', '\r\n']:
                    word_ids = self.__create_word_ids(bucket)
                    self.sentence_word_ids.append(word_ids)
                    wordchr_ids = self.__create_wordchr_ids(bucket)
                    self.sentence_wordchr_ids.append(wordchr_ids)
                    pos_ids = self.__create_pos_ids(bucket)
                    self.sentence_pos_ids.append(pos_ids)
                    etc, tag = self.__create_etc_and_tag(bucket)
                    self.sentence_etcs.append(etc)
                    self.sentence_tags.append(tag)
                    bucket = []
                else:
                    bucket.append(line)

    def __create_word_ids(self, bucket):
        """Create an word id vector
        """
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
        """Create a vector of a character id vector
        """
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

    def __create_pos_ids(self, bucket):
        """Create a pos id vector
        """
        pos_ids = []
        sentence_length = 0
        for line in bucket:
            line = line.strip()
            tokens = line.split()
            assert (len(tokens) == 4)
            sentence_length += 1
            pos = tokens[1]
            pid = self.config.embvec.get_pid(pos)
            pos_ids.append(pid)
            if sentence_length == self.max_sentence_length: break
        # padding with pad pid
        for _ in range(self.max_sentence_length - sentence_length):
            pos_ids.append(self.config.embvec.pad_pid)
        return pos_ids

    def __create_etc_and_tag(self, bucket):
        """Create a vector of a etc vector and an one-hot tag vector
        """
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
            temp = self.__shape_vec(tokens[0])                              # adding shape vec(5)
            temp = np.append(temp, self.__pos_vec(tokens[1]))               # adding pos one-hot(5)
            '''
            temp = np.append(temp, self.__chunk_vec(tokens[2]))             # adding chunk one-hot(5)
            temp = np.append(temp, tokens[4])                               # adding gazetteer feature
            '''
            etc.append(temp)
            tag.append(self.__tag_vec(tokens[3], self.config.class_size))   # tag one-hot(9)
            if sentence_length == self.max_sentence_length: break
        # padding with 0s
        for _ in range(self.max_sentence_length - sentence_length):
            temp = np.array([0 for _ in range(self.config.etc_dim)])
            etc.append(temp)
            tag.append(np.array([0] * self.config.class_size))
        return etc, tag

    def __pos_vec(self, t):
        """Build one-hot for pos

        build language specific features
        """
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

    def __chunk_vec(self, t):
        """Build one-hot for chunk

        build language specific features
        """
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

    def __shape_vec(self, word):
        """Build shape vector

        build language specific features:
          no-info[0], allDigits[1], mixedDigits[2], allSymbols[3],
          mixedSymbols[4], upperInitial[5], lowercase[6], allCaps[7], mixedCaps[8]
        """

        def is_capital(ch):
            if ord('A') <= ord(ch) <= ord('Z'): return True
            return False

        def is_symbol(ch):
            if not ch.isalpha() and not ch.isdigit() : return True
            return False

        one_hot = np.zeros(9)
        size = len(word)
        if word.isdigit():
            one_hot[1] = 1                            # allDigits
        elif word.isalpha():
            n_caps = 0
            for i in range(size):
                if is_capital(word[i]): n_caps += 1
            if n_caps == 0:
                one_hot[6] = 1                        # lowercase
            else:
                if size == n_caps: 
                    one_hot[7] = 1                    # allCaps
                else:
                    if is_capital(word[0]): 
                        one_hot[5] = 1                # upperInitial
                    else:
                        one_hot[8] = 1                # mixedCaps
        else:
            n_digits = 0
            n_symbols = 0
            for i in range(size):
                if word[i].isdigit(): n_digits += 1
                if is_symbol(word[i]): n_symbols += 1
            if n_digits >= 1: one_hot[2] = 1          # mixedDigits
            if n_symbols > 0:
                if size == n_symbols: one_hot[3] = 1  # allSymbols
                else: one_hot[4] = 1                  # mixedSymbols
            if n_digits == 0 and n_symbols == 0:
                one_hot[0] = 1                        # no-info
        return one_hot

    def __tag_vec(self, tag, class_size):
        """Build one-hot for tag
        """
        one_hot = np.zeros(class_size)
        tid = self.config.embvec.get_tid(tag)
        one_hot[tid] = 1
        return one_hot

    def logit_to_tags(self, logit, length):
        """Convert logit to tags

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
            tag = self.config.embvec.get_tag(tid)
            tags.append(tag)
        return tags

    def logit_indices_to_tags(self, logit_indices, length):
        """Convert logit_indices to tags

        Args:
          logit_indices: [sentence_length]
          length: int

        Returns:
          tag sequence(size length)
        """
        pred_list = logit_indices[0:length]
        tags = []
        for tid in pred_list:
            tag = self.config.embvec.get_tag(tid)
            tags.append(tag)
        return tags

    def logits_indices_to_tags_seq(self, logits_indices, lengths):
        """Convert logits_indices to tags sequence

        Args:
          logits_indices: [batch_size, sentence_length]
          lengths: [batch_size]

        Returns:
          sequence of tag sequence
        """
        tags_seq = []
        for logit_indices, length in zip(logits_indices, lengths):
            tags = self.logit_indices_to_tags(logit_indices, length)
            tags_seq.append(tags)
        return tags_seq

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

