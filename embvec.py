from __future__ import print_function
import sys
import argparse
import numpy as np
import pickle as pkl
from random import random

class EmbVec:

    def __init__(self, args):
        """Build embedding, vocabularies, other resources

        Args:
          args: args from this script(embvec.py).
        """
        self.pad = '#PAD#'
        self.unk = '#UNK#'
        self.lowercase = True
        if args.lowercase == 'False': self.lowercase = False

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

        self.chk_vocab = {}      # chunk vocab
        self.pad_kid = 0         # for padding chunk embedding
        self.unk_kid = 1         # for unknown chunk
        self.chk_vocab[self.pad] = self.pad_kid
        self.chk_vocab[self.unk] = self.unk_kid

        self.oot_tid = 0         # out of tag id
        self.oot_tag = 'O'       # out of tag, this is fixed for convenience
        self.xot_tid = 1         # 'X' tag id
        self.xot_tag = 'X'       # 'X' tag, fixed for convenience
        self.tag_vocab = {}      # tag vocab (tag -> id)
        self.itag_vocab = {}     # inverse tag vocab (id -> tag)
        self.tag_vocab[self.oot_tag] = self.oot_tid
        self.itag_vocab[0] = self.oot_tag
        self.tag_vocab[self.xot_tag] = self.xot_tid
        self.itag_vocab[1] = self.xot_tag
    
        self.wrd_vocab_tmp = {}  # word vocab for train/dev/test

        # elmo
        self.elmo_vocab = {}     # elmo vocab
        self.elmo_vocab_path   = args.elmo_vocab_path
        self.elmo_options_path = args.elmo_options_path
        self.elmo_weight_path  = args.elmo_weight_path

        # bert
        self.bert_config_path = args.bert_config_path
        self.bert_vocab_path  = args.bert_vocab_path
        self.bert_do_lower_case = False
        if args.bert_do_lower_case == 'True': self.bert_do_lower_case = True 
        self.bert_init_checkpoint = args.bert_init_checkpoint
        self.bert_max_seq_length = args.bert_max_seq_length
        self.bert_dim = args.bert_dim

        # build character/pos/chunk/tag/elmo vocab.
        cid = self.unk_cid + 1
        pid = self.unk_pid + 1
        kid = self.unk_kid + 1
        tid = self.xot_tid + 1
        for line in open(args.train_path):
            line = line.strip()
            if not line: continue
            tokens = line.split()
            assert(len(tokens) == 4)
            word = tokens[0]
            pos  = tokens[1]
            chk = tokens[2]
            tag  = tokens[3]
            # character vocab
            for ch in word:
                if ch not in self.chr_vocab:
                    self.chr_vocab[ch] = cid
                    cid += 1
            # elmo vocab(case sensitive)
            if word not in self.elmo_vocab: self.elmo_vocab[word] = 1
            else: self.elmo_vocab[word] += 1
            # pos vocab
            if pos not in self.pos_vocab:
                self.pos_vocab[pos] = pid
                pid += 1
            # chunk vocab
            if chk not in self.chk_vocab:
                self.chk_vocab[chk] = kid
                kid += 1
            # tag, itag vocab
            if tag not in self.tag_vocab:
                self.tag_vocab[tag] = tid
                self.itag_vocab[tid] = tag
                tid += 1
            if self.lowercase: word = word.lower()
            if word not in self.wrd_vocab_tmp:
                self.wrd_vocab_tmp[word] = 0
        # write elmo vocab.
        if self.elmo_vocab_path:
            elmo_vocab_fd = open(self.elmo_vocab_path, 'w')
            elmo_vocab_fd.write('<S>' + '\n')
            elmo_vocab_fd.write('</S>' + '\n')
            elmo_vocab_fd.write('<UNK>' + '\n')
            for word, freq in sorted(self.elmo_vocab.items(), key=lambda x: x[1], reverse=True):
                elmo_vocab_fd.write(word + '\n')
            elmo_vocab_fd.close()
        del(self.elmo_vocab)

        # build word embeddings and word vocab.
        wrd_vocab_size = 0
        for line in open(args.emb_path): wrd_vocab_size += 1
        wrd_vocab_size += 2 # for pad, unk
        sys.stderr.write('wrd_vocab_size = %s\n' % (wrd_vocab_size))
        self.wrd_dim = args.wrd_dim
        self.wrd_embeddings = np.zeros((wrd_vocab_size, self.wrd_dim))
        # 0 id for padding
        vector = np.array([float(0) for i in range(self.wrd_dim)])
        self.wrd_embeddings[self.pad_wid] = vector
        # 1 wid for unknown
        vector = np.array([random() for i in range(self.wrd_dim)])
        self.wrd_embeddings[self.unk_wid] = vector
        wid = self.unk_wid + 1
        for line in open(args.emb_path):
            line = line.strip()
            tokens = line.split()
            word = tokens[0]
            try: vector = np.array([float(val) for val in tokens[1:]])
            except: continue
            if len(vector) != self.wrd_dim: continue
            if self.lowercase: word = word.lower()
            self.wrd_embeddings[wid] = vector
            self.wrd_vocab[word] = wid
            wid += 1
        del(self.wrd_vocab_tmp)

    def get_wid(self, word):
        if self.lowercase: word = word.lower()
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

    def get_kid(self, chk):
        if chk in self.chk_vocab:
            return self.chk_vocab[chk]
        return self.unk_kid

    def get_tid(self, tag):
        if tag in self.tag_vocab:
            return self.tag_vocab[tag]
        return self.oot_tid

    def get_tag(self, tid):
        if tid in self.itag_vocab:
            return self.itag_vocab[tid]
        return self.oot_tag
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str, help='path to a file of word embedding vector(.txt)', required=True)
    parser.add_argument('--wrd_dim', type=int, help='embedding vector dimension', required=True)
    parser.add_argument('--train_path', type=str, help='path to a train file', required=True)
    parser.add_argument('--lowercase', type=str, help='apply lower case for word embedding', default=True)
    parser.add_argument('--elmo_vocab_path', type=str, help='path to elmo vocab file(write)', default='')
    parser.add_argument('--elmo_options_path', type=str, help='path to elmo options file', default='')
    parser.add_argument('--elmo_weight_path', type=str, help='path to elmo weight file', default='')
    parser.add_argument('--bert_config_path', type=str, help='path to bert config file', default='')
    parser.add_argument('--bert_vocab_path', type=str, help='path to bert vocab file', default='')
    parser.add_argument('--bert_do_lower_case', type=str, help='apply lower case for bert', default=False)
    parser.add_argument('--bert_init_checkpoint', type=str, help='path to bert init checkpoint', default='')
    parser.add_argument('--bert_max_seq_length', type=int, help='maximum total input sequence length after WordPiece tokenization.', default=180)
    parser.add_argument('--bert_dim', type=int, help='bert output dimension size', default=1024)
    args = parser.parse_args()
    embvec = EmbVec(args)
    pkl.dump(embvec, open(args.emb_path + '.pkl', 'wb'))

    # print all vocab for inference by C++.
    # 1. wrd_vocab
    print('# wrd_vocab', len(embvec.wrd_vocab))
    for word, wid in embvec.wrd_vocab.items():
        print(word, wid)
    # 2. chr_vocab
    print('# chr_vocab', len(embvec.chr_vocab))
    for ch, cid in embvec.chr_vocab.items():
        print(ch, cid)
    # 3. pos_vocab
    print('# pos_vocab', len(embvec.pos_vocab))
    for pos, pid in embvec.pos_vocab.items():
        print(pos, pid)
    # 4. chk_vocab
    print('# chk_vocab', len(embvec.chk_vocab))
    for chk, kid in embvec.chk_vocab.items():
        print(chk, kid)
    # 5. tag_vocab
    print('# tag_vocab', len(embvec.tag_vocab))
    for tag, tid, in embvec.tag_vocab.items():
        print(tag, tid)
