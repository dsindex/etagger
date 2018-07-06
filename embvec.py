from __future__ import print_function
import numpy as np
import pickle as pkl
from randomvec import RandomVec

import argparse

class EmbVec:
    def __init__(self, args):
        self.model = {}
        self.rand_model = RandomVec(args.emb_dim)
        path = args.emb_path
        invalid = 0
        for line in open(path):
            line = line.split()
            word = line[0]
            vector = np.array([float(val) for val in line[1:]])
            if len(vector) != args.emb_dim:
                invalid += 1
                continue
            self.model[word] = vector
        print('invalid entries %d' % invalid)

    def __getitem__(self, word):
        word = word.lower()
        try:
            return self.model[word]
        except KeyError:
            return self.rand_model[word]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str, help='path to word embedding vector(.txt)', required=True)
    parser.add_argument('--emb_dim', type=int, help='vector dimension', required=True)
    args = parser.parse_args()
    embvec = EmbVec(args)
    pkl.dump(embvec, open(args.emb_path + '.pkl', 'wb'))
