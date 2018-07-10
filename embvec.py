import numpy as np
import pickle as pkl
from random import random
import sys
import argparse

class EmbVec:
    def __init__(self, args):
        self.model = {}
        self.dim = args.emb_dim
        invalid = 0
        for line in open(args.emb_path):
            line = line.split()
            word = line[0]
            vector = np.array([float(val) for val in line[1:]])
            if len(vector) != self.dim:
                invalid += 1
                continue
            self.model[word] = vector
        sys.stderr.write('invalid entries %d' % (invalid) + '\n')

    def __getitem__(self, word):
        word = word.lower()
        try:
            return self.model[word]
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
