from __future__ import print_function
import numpy as np
import pickle as pkl
from random import random
import sys
import argparse

class EmbVec:
    def __init__(self, args):
        self.pad = '#PAD#'
        self.unk = '#UNK#'
        self.wvocab = {}      # word vocab
        self.embeddings = []
        self.wrd_dim = args.wrd_dim
        self.pad_wid = 0      # for padding word embedding
        self.unk_wid = 1      # for unknown word
        self.wvocab[self.pad] = self.pad_wid
        self.wvocab[self.unk] = self.unk_wid
        self.cvocab = {}      # character vocab
        self.pad_cid = 0      # for padding char embedding
        self.unk_cid = 1      # for unknown char
        self.cvocab[self.pad] = self.pad_cid
        self.cvocab[self.unk] = self.unk_cid
        self.oot_tid = 0      # out of tag id
        self.oot_tag = 'O'    # out of tag, this is fixed for convenience
        self.tag_vocab = {}   # tag vocab (tag -> id)
        self.itag_vocab = {}  # inverse tag vocab (id -> tag)
        self.tag_vocab[self.oot_tag] = self.oot_tid
        self.itag_vocab[0] = self.oot_tag
        self.gaz_vocab = {}   # gazetteer vocab

        invalid = 0
        # 0 id for padding
        vector = np.array([float(0) for i in range(self.wrd_dim)])
        assert(len(vector) == self.wrd_dim)
        self.embeddings.append(vector)
        # 1 wid for unknown
        vector = np.array([random() for i in range(self.wrd_dim)])
        assert(len(vector) == self.wrd_dim)
        self.embeddings.append(vector)
        # 2 wid ~ for normal entries
        wid = self.unk_wid + 1
        for line in open(args.emb_path):
            line = line.strip()
            line = line.split()
            word = line[0].lower()
            vector = np.array([float(val) for val in line[1:]])
            if len(vector) != self.wrd_dim:
                invalid += 1
                continue
            self.wvocab[word] = wid
            self.embeddings.append(vector)
            wid += 1
        sys.stderr.write('invalid entries %d' % (invalid) + '\n')
        # 2 cid ~ for normal characters
        cid = self.unk_cid + 1
        tid = self.oot_tid + 1
        for line in open(args.train_path):
            tokens = line.split()
            if len(tokens) != 4: continue
            word = tokens[0]
            tag  = tokens[3]
            # character vocab
            for ch in word:
                if ch not in self.cvocab:
                    self.cvocab[ch] = cid
                    cid += 1
            # tag, itag vocab
            if tag not in self.tag_vocab:
                self.tag_vocab[tag] = tid
                self.itag_vocab[tid] = tag
                tid += 1
        # build gazetteer features
        for line in open(args.train_path):
            line = line.strip()
            tokens = line.split()
            if len(tokens) != 4: continue
            word = tokens[0].lower()
            tag  = tokens[3]
            # filtering
            if word[0].isdigit(): continue
            if not word[0].isalpha(): continue
            if len(word) <= 2: continue
            '''
            # 1|0 setting
            if tag == self.oot_tag: continue
            if word not in self.gaz_vocab:
                self.gaz_vocab[word] = np.zeros(1)
                self.gaz_vocab[word][0] = 1
            '''
            # m-hot setting
            if word in self.gaz_vocab:
                gaz = self.gaz_vocab[word]
                tid = self.tag_vocab[tag]
                gaz[tid] = 1
            else:
                self.gaz_vocab[word] = np.zeros(len(self.tag_vocab))
                tid = self.tag_vocab[tag]
                self.gaz_vocab[word][tid] = 1
        
    def get_wid(self, word):
        word = word.lower()
        if word in self.wvocab:
            return self.wvocab[word]
        return self.unk_wid

    def get_cid(self, ch):
        ch = ch.lower()
        if ch in self.cvocab:
            return self.cvocab[ch]
        return self.unk_cid

    def get_tid(self, tag):
        if tag in self.tag_vocab:
            return self.tag_vocab[tag]
        return self.oot_tid

    def get_tag(self, tid):
        if tid in self.itag_vocab:
            return self.itag_vocab[tid]
        return self.oot_tag

    def get_gaz(self, word):
        '''
        # 0|1 setting
        word = word.lower()
        if word in self.gaz_vocab:
            return self.gaz_vocab[word]
        return np.zeros(1)
        '''
        # m-hot setting
        word = word.lower()
        if word in self.gaz_vocab:
            # check ambiguity
            vec = self.gaz_vocab[word]
            count = 0
            for i in range(len(vec)):
                if vec[i]: count += 1
            if count >= 2: return np.zeros(len(self.tag_vocab))
            return vec
        return np.zeros(len(self.tag_vocab))

    def __getitem__(self, wid):
        try:
            return self.embeddings[wid]
        except KeyError:
            vec = np.array([random() for i in range(self.wrd_dim)])
            return vec

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str, help='path to a file of word embedding vector(.txt)', required=True)
    parser.add_argument('--wrd_dim', type=int, help='embedding vector dimension', required=True)
    parser.add_argument('--train_path', type=str, help='path to a training file', required=True)
    args = parser.parse_args()
    embvec = EmbVec(args)
    pkl.dump(embvec, open(args.emb_path + '.pkl', 'wb'))
    '''
    for word, _ in embvec.gaz_vocab.iteritems():
        print(word, embvec.get_gaz(word))
    '''
