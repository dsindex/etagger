from __future__ import print_function
import numpy as np
import pickle
from embvec import EmbVec
import sys


class Input:

    def __init__(self, data, embvec, emb_dim, class_size, sentence_length=-1):
        self.sentence = []
        self.sentence_tag = []
        self.embvec = embvec
        self.emb_dim = emb_dim
        self.class_size = class_size
        if sentence_length == -1:
            if type(data) is list:
                self.max_sentence_length = len(data)
            else: # treat as file path
                self.max_sentence_length = self.find_max_length(data)
        else:
            self.max_sentence_length = sentence_length
        # 'emb_dim + (5+5+1)' number of 0's
        self.word_dim = emb_dim + 11

        if type(data) is list:
            word, tag = self.__create_input(data)
            self.sentence.append(word)
            self.sentence_tag.append(tag)
        else: # treat as file path
            bucket = []
            for line in open(data):
                if line in ['\n', '\r\n']:
                    word, tag = self.__create_input(bucket)
                    self.sentence.append(word)
                    self.sentence_tag.append(tag)
                    bucket = []
                else:
                    bucket.append(line)

    def __create_input(self, bucket):
        word = []
        tag  = []
        sentence_length = 0
        for line in bucket:
            tokens = line.split()
            assert (len(tokens) == 4)
            sentence_length += 1
            temp = self.embvec[tokens[0]]
            assert len(temp) == self.emb_dim
            temp = np.append(temp, self.pos(tokens[1]))        # adding pos one-hot(5)
            temp = np.append(temp, self.chunk(tokens[2]))      # adding chunk one-hot(5)
            temp = np.append(temp, self.capital(tokens[0]))    # adding capital one-hot(1)
            word.append(temp)
            tag.append(self.label(tokens[3], self.class_size)) # label one-hot(9)
        # padding
        for _ in range(self.max_sentence_length - sentence_length):
            # nine 0's
            tag.append(np.array([0] * self.class_size))
            temp = np.array([0 for _ in range(self.word_dim)])
            word.append(temp)
        return word, tag

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
        if ord('A') <= ord(word[0]) <= ord('Z'):
            return np.array([1])
        else:
            return np.array([0])

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
        pred : [args.senence_length, args.class_size]
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

