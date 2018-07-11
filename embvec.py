import numpy as np
import pickle as pkl
from random import random
import sys
import argparse

class EmbVec:
    def __init__(self, args):
        self.vocab = {}
        self.embeddings = []
        self.dim = args.emb_dim
        self.pad_id = 0  # for padding
        self.unk_id = 1  # for unknown
        invalid = 0
        # 0 id for padding
        vector = np.array([float(0) for i in range(self.dim)])
        assert(len(vector) == self.dim)
        self.embeddings.append(vector)
        # 1 id for unknown
        vector = np.array([random() for i in range(self.dim)])
        assert(len(vector) == self.dim)
        self.embeddings.append(vector)
        # 2 id ~ 
        id = self.unk_id + 1
        for line in open(args.emb_path):
            line = line.split()
            word = line[0].lower()
            vector = np.array([float(val) for val in line[1:]])
            if len(vector) != self.dim:
                invalid += 1
                continue
            self.vocab[word] = id
            self.embeddings.append(vector)
            id += 1
        sys.stderr.write('invalid entries %d' % (invalid) + '\n')

    def get_id(self, word):
        word = word.lower()
        if word in self.vocab:
            return self.vocab[word]
        return self.unk_id

    def __getitem__(self, id):
        try:
            return self.embeddings[id]
        except KeyError:
            vec = np.array([random() for i in range(self.dim)])
            return vec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str, help='path to word embedding vector(.txt)', required=True)
    parser.add_argument('--emb_dim', type=int, help='vector dimension', required=True)
    args = parser.parse_args()
    embvec = EmbVec(args)
    pkl.dump(embvec, open(args.emb_path + '.pkl', 'wb'))
