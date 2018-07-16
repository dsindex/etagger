import numpy as np
import pickle as pkl
from random import random
import sys
import argparse

class EmbVec:
    def __init__(self, args):
        self.pad = '#PAD#'
        self.unk = '#UNK#'
        self.wvocab = {}
        self.embeddings = []
        self.wrd_dim = args.wrd_dim
        self.pad_wid = 0  # for padding word embedding
        self.unk_wid = 1  # for unknown word
        self.wvocab[self.pad] = self.pad_wid
        self.wvocab[self.unk] = self.unk_wid
        self.cvocab = {}
        self.pad_cid = 0  # for padding char embedding
        self.unk_cid = 1  # for unknown char
        self.cvocab[self.pad] = self.pad_cid
        self.cvocab[self.unk] = self.unk_cid

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
        for line in open(args.train_path):
            tokens = line.split()
            if len(tokens) != 4: continue
            word = tokens[0]
            for ch in word:
                if ch not in self.cvocab:
                    self.cvocab[ch] = cid
                    cid += 1

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
