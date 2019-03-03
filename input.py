from __future__ import print_function
import sys
import re
import tensorflow as tf
import numpy as np
from embvec import EmbVec

class Input:
    def __init__(self, data, config, build_output=True):

        # compute max sentence length
        if type(data) is list:
            self.max_sentence_length = len(data)
        else: # treat as file path
            self.max_sentence_length = self.find_max_length(data)

        if config.emb_class == 'bert':
            self.max_sentence_length = config.bert_max_seq_length # NOTE trick for reusing codes.
            self.sentence_word_ids = []                    # [batch_size, bert_max_seq_length]
            self.sentence_wordchr_ids = []                 # [batch_size, bert_max_seq_length, word_length]
            self.sentence_pos_ids = []                     # [batch_size, bert_max_seq_length]
            self.sentence_chk_ids = []                   # [batch_size, bert_max_seq_length]
            self.sentence_bert_token_ids = []              # [batch_size, bert_max_seq_length]
            self.sentence_bert_token_masks = []            # [batch_size, bert_max_seq_length]
            self.sentence_bert_segment_ids = []            # [batch_size, bert_max_seq_length]
            self.sentence_bert_wordidx2tokenidx = []       # [batch_size]
            if build_output:
                self.sentence_tags = []                    # [batch_size, bert_max_seq_length, class_size] 
        else:
            self.sentence_word_ids = []                    # [batch_size, max_sentence_length]
            self.sentence_wordchr_ids = []                 # [batch_size, max_sentence_length, word_length]
            if config.emb_class == 'elmo':
                self.sentence_elmo_wordchr_ids = []        # [batch_size, max_sentence_length+2, word_length]
            self.sentence_pos_ids = []                     # [batch_size, max_sentence_length]
            self.sentence_chk_ids = []                     # [batch_size, max_sentence_length]
            if build_output:
                self.sentence_tags = []                    # [batch_size, max_sentence_length, class_size] 
        self.config = config


        if type(data) is list: # treat data as bucket
            bucket = data
            ex_index = 0
            if config.emb_class == 'bert':
                bert_token_ids, bert_token_masks, bert_segment_ids, \
                bert_word_ids, bert_wordchr_ids, bert_pos_ids, bert_chk_ids, bert_tag, bert_wordidx2tokenidx = \
                    self.__create_bert_input(bucket, ex_index)
                self.sentence_word_ids.append(bert_word_ids)
                self.sentence_wordchr_ids.append(bert_wordchr_ids)
                self.sentence_pos_ids.append(bert_pos_ids)
                self.sentence_chk_ids.append(bert_chk_ids)
                self.sentence_bert_token_ids.append(bert_token_ids)
                self.sentence_bert_token_masks.append(bert_token_masks)
                self.sentence_bert_segment_ids.append(bert_segment_ids)
                self.sentence_bert_wordidx2tokenidx.append(bert_wordidx2tokenidx)
                if build_output:
                    self.sentence_tags.append(bert_tag)
            else:
                word_ids = self.__create_word_ids(bucket)
                self.sentence_word_ids.append(word_ids)
                wordchr_ids = self.__create_wordchr_ids(bucket)
                self.sentence_wordchr_ids.append(wordchr_ids)
                if config.emb_class == 'elmo':
                    elmo_wordchr_ids = self.__create_elmo_wordchr_ids(bucket)
                    self.sentence_elmo_wordchr_ids.append(elmo_wordchr_ids)
                pos_ids = self.__create_pos_ids(bucket)
                self.sentence_pos_ids.append(pos_ids)
                chk_ids = self.__create_chk_ids(bucket)
                self.sentence_chk_ids.append(chk_ids)
                if build_output:
                    tag = self.__create_tag(bucket)
                    self.sentence_tags.append(tag)
        else:                  # treat data as file path
            path = data
            bucket = []
            ex_index = 0
            for line in open(path):
                if line in ['\n', '\r\n']:
                    if config.emb_class == 'bert':
                        bert_token_ids, bert_token_masks, bert_segment_ids, \
                        bert_word_ids, bert_wordchr_ids, bert_pos_ids, bert_chk_ids, bert_tag, bert_wordidx2tokenidx = \
                            self.__create_bert_input(bucket, ex_index)
                        self.sentence_word_ids.append(bert_word_ids)
                        self.sentence_wordchr_ids.append(bert_wordchr_ids)
                        self.sentence_pos_ids.append(bert_pos_ids)
                        self.sentence_chk_ids.append(bert_chk_ids)
                        self.sentence_bert_token_ids.append(bert_token_ids)
                        self.sentence_bert_token_masks.append(bert_token_masks)
                        self.sentence_bert_segment_ids.append(bert_segment_ids)
                        self.sentence_bert_wordidx2tokenidx.append(bert_wordidx2tokenidx)
                        if build_output:
                            self.sentence_tags.append(bert_tag)
                    else:
                        word_ids = self.__create_word_ids(bucket)
                        self.sentence_word_ids.append(word_ids)
                        wordchr_ids = self.__create_wordchr_ids(bucket)
                        self.sentence_wordchr_ids.append(wordchr_ids)
                        if config.emb_class == 'elmo':
                            elmo_wordchr_ids = self.__create_elmo_wordchr_ids(bucket)
                            self.sentence_elmo_wordchr_ids.append(elmo_wordchr_ids)
                        pos_ids = self.__create_pos_ids(bucket)
                        self.sentence_pos_ids.append(pos_ids)
                        chk_ids = self.__create_chk_ids(bucket)
                        self.sentence_chk_ids.append(chk_ids)
                        if build_output:
                            tag = self.__create_tag(bucket)
                            self.sentence_tags.append(tag)
                    bucket = []
                    ex_index += 1
                else:
                    bucket.append(line)

    def __create_bert_input(self, bucket, ex_index):
        """Create a vector of
               bert token id,
               bert token mask,
               bert segment id,
               bert word id,
               bert wordchr id,
               bert pos id,
               bert chk id,
               bert tag
               bert wordidx 2 tokenidx,
        """
        word_ids = self.__create_word_ids(bucket)
        wordchr_ids = self.__create_wordchr_ids(bucket)
        pos_ids = self.__create_pos_ids(bucket)
        chk_ids = self.__create_chk_ids(bucket)
        tag = self.__create_tag(bucket)

        bert_word_ids = []
        bert_wordchr_ids = []
        bert_pos_ids = []
        bert_chk_ids = []
        bert_tag = []

        bert_tokenizer = self.config.bert_tokenizer
        bert_max_seq_length = self.config.bert_max_seq_length
        ntokens = []
        bert_segment_ids = []
        bert_wordidx2tokenidx = {}

        ntokens.append('[CLS]')
        ntokens_last = 0
        bert_segment_ids.append(0)
        bert_word_ids.append(self.config.embvec.pad_wid) # 0
        pad_chr_ids = []
        for _ in range(self.config.word_length):
            pad_chr_ids.append(self.config.embvec.pad_cid) # 0
        bert_wordchr_ids.append(pad_chr_ids)
        bert_pos_ids.append(self.config.embvec.unk_pid) # 1, do not use pad_pid
        bert_chk_ids.append(self.config.embvec.unk_kid) # 1, unk_kid
        bert_tag.append(self.__tag_vec(self.config.embvec.oot_tag, self.config.class_size)) # 'O' tag

        for i, line in enumerate(bucket):
            line = line.strip()
            tokens = line.split()
            assert (len(tokens) == 4)
            word = tokens[0]
            bert_tokens = bert_tokenizer.tokenize(word)
            for j, bert_token in enumerate(bert_tokens):
                ntokens.append(bert_token)
                ntokens_last += 1
                bert_segment_ids.append(0)
                # extend bert_word_ids, bert_wordchr_ids, bert_pos_ids, bert_chk_ids, bert_tag
                bert_word_ids.append(word_ids[i])
                bert_wordchr_ids.append(wordchr_ids[i])
                bert_pos_ids.append(pos_ids[i])
                bert_chk_ids.append(chk_ids[i])
                if j == 0:
                    bert_tag.append(tag[i])
                    bert_wordidx2tokenidx[i] = ntokens_last
                else:
                    bert_tag.append(self.__tag_vec(self.config.embvec.xot_tag, self.config.class_size)) # 'X' tag
            if len(ntokens) == bert_max_seq_length - 1:
                tf.logging.info('len(ntokens): %s' % str(len(ntokens)))
                break
        '''
        ntokens.append('[SEP]')
        ntokens_last += 1
        bert_segment_ids.append(0)
        bert_word_ids.append(self.config.embvec.pad_wid) # 0
        bert_wordchr_ids.append(pad_chr_ids)
        bert_pos_ids.append(self.config.embvec.unk_pid) # 1, do not use pad_pid
        bert_chk_ids.append(self.config.embvec.unk_kid) # 1, unk_kid
        bert_tag.append(self.__tag_vec(self.config.embvec.oot_tag, self.config.class_size)) # 'O' tag
        '''

        bert_token_ids = bert_tokenizer.convert_tokens_to_ids(ntokens)
        bert_token_masks = [1] * len(bert_token_ids)

        # padding for bert_token_ids, bert_token_masks, bert_segment_ids
        while len(bert_token_ids) < bert_max_seq_length:
            bert_token_ids.append(0)
            bert_token_masks.append(0)
            bert_segment_ids.append(0)
        assert len(bert_token_ids) == bert_max_seq_length
        assert len(bert_token_masks) == bert_max_seq_length
        assert len(bert_segment_ids) == bert_max_seq_length
        # padding for bert_word_ids, bert_wordchr_ids, bert_pos_ids, bert_chk_ids, bert_tag
        while len(bert_word_ids) < bert_max_seq_length:
            bert_word_ids.append(self.config.embvec.pad_wid)
            bert_wordchr_ids.append(pad_chr_ids)
            bert_pos_ids.append(self.config.embvec.pad_pid)
            bert_chk_ids.append(self.config.embvec.pad_kid)
            bert_tag.append(np.array([0] * self.config.class_size))
        assert len(bert_word_ids) == bert_max_seq_length
        assert len(bert_wordchr_ids) == bert_max_seq_length
        assert len(bert_pos_ids) == bert_max_seq_length
        assert len(bert_chk_ids) == bert_max_seq_length
        assert len(bert_tag) == bert_max_seq_length

        if ex_index < 5:
            from bert import tokenization  
            tf.logging.info('*** Example ***')
            tf.logging.info('ntokens: %s' % ' '.join([tokenization.printable_text(x) for x in ntokens]))
            tf.logging.info('bert_token_ids: %s' % ' '.join([str(x) for x in bert_token_ids]))
            tf.logging.info('bert_token_masks: %s' % ' '.join([str(x) for x in bert_token_masks]))
            tf.logging.info('bert_segment_ids: %s' % ' '.join([str(x) for x in bert_segment_ids]))
            tf.logging.info('bert_word_ids: %s' % ' '.join([str(x) for x in bert_word_ids]))
            '''
            tf.logging.info('bert_wordchr_ids: %s' % ' '.join([str(x) for x in bert_wordchr_ids]))
            tf.logging.info('bert_pos_ids: %s' % ' '.join([str(x) for x in bert_pos_ids]))
            tf.logging.info('bert_chk_ids: %s' % ' '.join([str(x) for x in bert_chk_ids]))
            tf.logging.info('bert_tag: %s' % ' '.join([str(x) for x in bert_tag]))
            '''

        return bert_token_ids, bert_token_masks, bert_segment_ids, bert_word_ids, bert_wordchr_ids, bert_pos_ids, bert_chk_ids, bert_tag, bert_wordidx2tokenidx

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
            word = tokens[0]
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
            word = tokens[0]
            for ch in list(word):
                word_length += 1 
                cid = self.config.embvec.get_cid(ch)
                chr_ids.append(cid)
                if word_length == self.config.word_length: break
            # padding with pad cid
            for _ in range(self.config.word_length - word_length):
                chr_ids.append(self.config.embvec.pad_cid)
            wordchr_ids.append(chr_ids)
            if sentence_length == self.max_sentence_length: break
        # padding with [pad_cid, ..., pad_cid] chr_ids
        for _ in range(self.max_sentence_length - sentence_length):
            pad_chr_ids = []
            for _ in range(self.config.word_length):
                pad_chr_ids.append(self.config.embvec.pad_cid)
            wordchr_ids.append(pad_chr_ids)
        return wordchr_ids

    def __create_elmo_wordchr_ids(self, bucket):
        """Create a vector of a character id vector for elmo
        """
        sentence = []
        sentence_length = 0
        for line in bucket:
            line = line.strip()
            tokens = line.split()
            assert (len(tokens) == 4)
            sentence_length += 1
            word = tokens[0]
            sentence.append(word)
            if sentence_length == self.max_sentence_length: break
        elmo_wordchr_ids = self.config.elmo_batcher.batch_sentences([sentence])[0].tolist()
        # padding with [0,...,0] chr_ids, '+2' stands for '<S>, </S>'
        for _ in range(self.max_sentence_length - len(elmo_wordchr_ids) + 2):
            chr_ids = []
            for _ in range(self.config.word_length):
                chr_ids.append(0)
            elmo_wordchr_ids.append(chr_ids)
        assert(len(elmo_wordchr_ids) == self.max_sentence_length+2)
        return elmo_wordchr_ids

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

    def __create_chk_ids(self, bucket):
        """Create a chk id vector
        """
        chk_ids = []
        sentence_length = 0
        for line in bucket:
            line = line.strip()
            tokens = line.split()
            assert (len(tokens) == 4)
            sentence_length += 1
            chk = tokens[2]
            kid = self.config.embvec.get_kid(chk)
            chk_ids.append(kid)
            if sentence_length == self.max_sentence_length: break
        # padding with pad kid
        for _ in range(self.max_sentence_length - sentence_length):
            chk_ids.append(self.config.embvec.pad_kid)
        return chk_ids

    def __create_tag(self, bucket):
        """Create a vector of an one-hot tag vector
        """
        tag  = []
        sentence_length = 0
        for line in bucket:
            line = line.strip()
            tokens = line.split()
            assert (len(tokens) == 4)
            sentence_length += 1
            tag.append(self.__tag_vec(tokens[3], self.config.class_size))   # tag one-hot
            if sentence_length == self.max_sentence_length: break
        # padding with 0s
        for _ in range(self.max_sentence_length - sentence_length):
            tag.append(np.array([0] * self.config.class_size))
        return tag

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
        """Convert logits_indices to sequence of tags

        Args:
          logits_indices: [batch_size, sentence_length]
          lengths: [batch_size]

        Returns:
          sequence of tags
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
