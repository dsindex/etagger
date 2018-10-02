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
        self.wrd_vocab = {}      # word vocab
        self.pad_wid = 0         # for padding word embedding
        self.unk_wid = 1         # for unknown word
        self.wrd_vocab[self.pad] = self.pad_wid
        self.wrd_vocab[self.unk] = self.unk_wid
        self.chr_vocab = {}      # character vocab
        self.pad_cid = 0         # for padding char embedding
        self.unk_cid = 1         # for unknown char
        self.chr_vocab[self.pad] = self.pad_cid
        self.chr_vocab[self.unk] = self.unk_cid
        self.pos_vocab = {}      # pos vocab
        self.pad_pid = 0         # for padding pos embedding
        self.unk_pid = 1         # for unknown pos
        self.pos_vocab[self.pad] = self.pad_pid
        self.pos_vocab[self.unk] = self.unk_pid
        self.oot_tid = 0         # out of tag id
        self.oot_tag = 'O'       # out of tag, this is fixed for convenience
        self.tag_vocab = {}      # tag vocab (tag -> id)
        self.itag_vocab = {}     # inverse tag vocab (id -> tag)
        self.tag_vocab[self.oot_tag] = self.oot_tid
        self.itag_vocab[0] = self.oot_tag
        self.tag_prefix_b = 'B-'
        self.tag_prefix_i = 'I-'
        self.gaz_vocab = {}      # gazetteer vocab

        # build word/character/pos/tag vocab
        wid = self.unk_wid + 1
        cid = self.unk_cid + 1
        pid = self.unk_pid + 1
        tid = self.oot_tid + 1
        for line in open(args.total_path):
            line = line.strip()
            if not line: continue
            tokens = line.split()
            assert(len(tokens) == 4)
            word = tokens[0]
            pos  = tokens[1]
            tag  = tokens[3]
            # word vocab
            if word not in self.wrd_vocab:
                self.wrd_vocab[word] = wid
                wid += 1
            # character vocab
            for ch in word:
                if ch not in self.chr_vocab:
                    self.chr_vocab[ch] = cid
                    cid += 1
            # pos vocab
            if pos not in self.pos_vocab:
                self.pos_vocab[pos] = pid
                pid += 1
            # tag, itag vocab
            if tag not in self.tag_vocab:
                self.tag_vocab[tag] = tid
                self.itag_vocab[tid] = tag
                tid += 1

        # build word embeddings
        wrd_vocab_size = len(self.wrd_vocab)
        self.wrd_dim = args.wrd_dim
        self.wrd_embeddings = np.zeros((wrd_vocab_size, self.wrd_dim))
        # 0 id for padding
        vector = np.array([float(0) for i in range(self.wrd_dim)])
        self.wrd_embeddings[self.pad_wid] = vector
        # 1 wid for unknown
        vector = np.array([random() for i in range(self.wrd_dim)])
        self.wrd_embeddings[self.unk_wid] = vector
        for line in open(args.emb_path):
            line = line.strip()
            tokens = line.split()
            word = tokens[0]
            try: vector = np.array([float(val) for val in tokens[1:]])
            except: continue
            if len(vector) != self.wrd_dim: continue
            # FIXME for fast training. when it comes to service, comment out
            if word not in self.wrd_vocab: continue
            wid = self.wrd_vocab[word]
            self.wrd_embeddings[wid] = vector

        # build gazetteer vocab
        bucket = []
        for line in open(args.train_path):
            if line in ['\n', '\r\n']:
                bucket_size = len(bucket)
                for i in range(bucket_size):
                    tokens = bucket[i]
                    word = tokens[0]
                    tag  = tokens[3]
                    if self.tag_prefix_b not in tag: continue
                    segment = self.__get_segment(bucket, bucket_size, i)
                    if not segment: continue
                    tag_suffix = tag.split('-')[1]
                    # noise filtering
                    if len(segment) <= 3: continue
                    if segment not in self.gaz_vocab:
                        self.gaz_vocab[segment] = {}
                        self.gaz_vocab[segment][tag_suffix] = 1
                    else:
                        self.gaz_vocab[segment][tag_suffix] = 1
                bucket = []
            else:
                line = line.strip()
                tokens = line.split()
                assert(len(tokens) == 4)
                bucket.append(tokens)
        
    def get_wid(self, word):
        if word in self.wrd_vocab:
            return self.wrd_vocab[word]
        return self.unk_wid

    def get_cid(self, ch):
        if ch in self.chr_vocab:
            return self.chr_vocab[ch]
        return self.unk_cid

    def get_pid(self, pos):
        if pos in self.pos_vocab:
            return self.pos_vocab[pos]
        return self.unk_pid

    def get_tid(self, tag):
        if tag in self.tag_vocab:
            return self.tag_vocab[tag]
        return self.oot_tid

    def get_tag(self, tid):
        if tid in self.itag_vocab:
            return self.itag_vocab[tid]
        return self.oot_tag

    def __get_segment(self, bucket, bucket_size, i):
        segment = []
        for j in range(i, bucket_size):
            tokens = bucket[j]
            word = tokens[0]
            tag  = tokens[3]
            valid = False
            if i == j : valid = True
            if i != j and self.tag_prefix_i in tag: valid = True
            if valid:
                segment.append(word)
            else: break
        return ''.join(segment)

    def get_gaz(self, word):
        if word in self.gaz_vocab:
            return self.gaz_vocab[word]
        return None

    def apply_gaz(self, bucket, bucket_size, i):
        # 1gram ~ 5gram
        for j in range(5, 0, -1): # max ngram size == 5
            if i+5 >= bucket_size: continue
            segment = []
            # bucket[i+0][0] ~ bucket[i+k][0]
            for k in range(j): segment.append(bucket[i+k][0])
            key = ''.join(segment)
            if key in self.gaz_vocab:
                for k in range(j):
                    gvec = bucket[i+k][4]
                    for tag_suffix, _ in self.gaz_vocab[key].items():
                        if k == 0: tag = self.tag_prefix_b + tag_suffix
                        else: tag = self.tag_prefix_i + tag_suffix
                        tid = self.get_tid(tag)
                        gvec[tid] = 1
                    '''
                    print(bucket[i+k])
                    '''
                # longest prefer
                return j
        return 0
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str, help='path to a file of word embedding vector(.txt)', required=True)
    parser.add_argument('--wrd_dim', type=int, help='embedding vector dimension', required=True)
    parser.add_argument('--train_path', type=str, help='path to a train file', required=True)
    parser.add_argument('--total_path', type=str, help='path to a train+dev+test file', required=True)
    args = parser.parse_args()
    embvec = EmbVec(args)
    pkl.dump(embvec, open(args.emb_path + '.pkl', 'wb'))
    '''
    # check gazetteer vocab
    for word, tags in embvec.gaz_vocab.items():
        print(word)
        for tag, _ in tags.items(): print(tag)
    '''
    # test before applying gazetteer feature
    bucket = []
    for line in open(args.train_path):
        if line in ['\n', '\r\n']:
            bucket_size = len(bucket)
            i = 0
            while 1:
                if i >= bucket_size: break
                tokens = bucket[i]
                j = embvec.apply_gaz(bucket, bucket_size, i)
                i += j # jump
                i += 1
            bucket = []
        else:
            line = line.strip()
            tokens = line.split()
            assert(len(tokens) == 4)
            gvec = np.zeros(len(embvec.tag_vocab))
            tokens.append(gvec)
            bucket.append(tokens)
