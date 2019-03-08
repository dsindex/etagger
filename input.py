from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
from embvec import EmbVec
import collections

class Input:

    def __init__(self, data, config, build_output=True, do_shuffle=False):
        self.config = config
        self.build_output = build_output

        if type(data) is list: # treat data as bucket
            # compute max sentence length
            self.max_sentence_length = len(data)
            self.num_examples = 1
            self.num_batches = 1
            # for inference, we use example directly
            self.example = None
            # create tf records
            self.__create_tfrecords(data)
        else:                  # treat as file path
            # compute max sentence length, number of examples, number of batches
            self.max_sentence_length, self.num_examples = self.stat(data)
            self.num_batches = (self.num_examples + config.batch_size - 1) // config.batch_size 
            # create tf records
            self.tfrecords_file = '.tfrecords' # init as suffix
            self.__create_tfrecords(data)
            # create dataset
            self.keys_to_features = self.__keys_to_features()
            self.dataset = self.__dataset_input_fn(config.batch_size, do_shuffle)
 
    def __create_tfrecords(self, data):
        """Create input tfrecords
        """

        # trick for reusing codes.
        if 'bert' in self.config.emb_class:
            self.max_sentence_length = self.config.bert_max_seq_length

        if type(data) is list: # treat data as bucket
            bucket = data
            ex_index = 0
            _, example = self.__create_single_tf_example(bucket, ex_index, is_inference=True)
            self.example = example 
        else:                  # treat data as file path
            path = data
            self.tfrecords_file = path + self.tfrecords_file
            writer = tf.python_io.TFRecordWriter(self.tfrecords_file)
            bucket = []
            ex_index = 0
            for line in open(path):
                if line in ['\n', '\r\n']:
                    tf_example, example = self.__create_single_tf_example(bucket, ex_index)
                    writer.write(tf_example.SerializeToString())
                    if ex_index % 500 == 0:
                        tf.logging.info("writing example %d" % (ex_index))
                    bucket = []
                    ex_index += 1
                else:
                    bucket.append(line)
            writer.close()

    def __keys_to_features(self):
        """Create keys to features map
        """
        keys_to_features = {}
        seq_length = self.max_sentence_length
        word_length = self.config.word_length
        class_size = self.config.class_size
        if 'bert' in self.config.emb_class:
            keys_to_features['word_ids'] = tf.FixedLenFeature([seq_length], tf.int64)
            keys_to_features['wordchr_ids'] = tf.FixedLenFeature([seq_length*word_length], tf.int64)
            keys_to_features['pos_ids'] = tf.FixedLenFeature([seq_length], tf.int64)
            keys_to_features['chk_ids'] = tf.FixedLenFeature([seq_length], tf.int64)
            keys_to_features['bert_token_ids'] = tf.FixedLenFeature([seq_length], tf.int64)
            keys_to_features['bert_token_masks'] = tf.FixedLenFeature([seq_length], tf.int64)
            keys_to_features['bert_segment_ids'] = tf.FixedLenFeature([seq_length], tf.int64)
            keys_to_features['bert_wordidx2tokenidx'] = tf.FixedLenFeature([seq_length], tf.int64)
            if 'elmo' in self.config.emb_class:
                keys_to_features['bert_elmo_indices'] = tf.FixedLenFeature([seq_length*2], tf.int64)
                keys_to_features['elmo_wordchr_ids'] = tf.FixedLenFeature([(seq_length+2)*word_length], tf.int64)
            if self.build_output:
                keys_to_features['tags'] = tf.FixedLenFeature([seq_length*class_size], tf.int64)
        else:
            keys_to_features['word_ids'] = tf.FixedLenFeature([seq_length], tf.int64)
            keys_to_features['wordchr_ids'] = tf.FixedLenFeature([seq_length*word_length], tf.int64)
            keys_to_features['pos_ids'] = tf.FixedLenFeature([seq_length], tf.int64)
            keys_to_features['chk_ids'] = tf.FixedLenFeature([seq_length], tf.int64)
            if 'elmo' in self.config.emb_class:
                keys_to_features['elmo_wordchr_ids'] = tf.FixedLenFeature([(seq_length+2)*word_length], tf.int64)
            if self.build_output:
                keys_to_features['tags'] = tf.FixedLenFeature([seq_length*class_size], tf.int64)
        return keys_to_features


    def __dataset_input_fn(self, batch_size, do_shuffle):
        """Build dataset input function
        """
        filenames = [self.tfrecords_file]
        dataset = tf.data.TFRecordDataset(filenames)

        def parser(record):
            parsed = tf.parse_single_example(record, self.keys_to_features)
            # convert 1D back to original dimension
            if 'bert' in self.config.emb_class:
                parsed['word_ids'] = tf.cast(parsed['word_ids'], tf.int32)
                parsed['wordchr_ids'] = tf.reshape(tf.cast(parsed['wordchr_ids'], tf.int32), [-1, self.config.word_length])
                parsed['pos_ids'] = tf.cast(parsed['pos_ids'], tf.int32)
                parsed['chk_ids'] = tf.cast(parsed['chk_ids'], tf.int32)
                parsed['bert_token_ids'] = tf.cast(parsed['bert_token_ids'], tf.int32)
                parsed['bert_token_masks'] = tf.cast(parsed['bert_token_masks'], tf.int32)
                parsed['bert_segment_ids'] = tf.cast(parsed['bert_segment_ids'], tf.int32)
                parsed['bert_wordidx2tokenidx'] = tf.cast(parsed['bert_wordidx2tokenidx'], tf.int32)
                if 'elmo' in self.config.emb_class:
                    parsed['bert_elmo_indices'] = tf.reshape(tf.cast(parsed['bert_elmo_indices'], tf.int32), [-1, 2])
                    parsed['elmo_wordchr_ids'] = tf.reshape(tf.cast(parsed['elmo_wordchr_ids'], tf.int32), [-1, self.config.word_length])
                if self.build_output:
                    parsed['tags'] = tf.reshape(tf.cast(parsed['tags'], tf.int32), [-1, self.config.class_size])
            else:
                parsed['word_ids'] = tf.cast(parsed['word_ids'], tf.int32)
                parsed['wordchr_ids'] = tf.reshape(tf.cast(parsed['wordchr_ids'], tf.int32), [-1, self.config.word_length])
                parsed['pos_ids'] = tf.cast(parsed['pos_ids'], tf.int32)
                parsed['chk_ids'] = tf.cast(parsed['chk_ids'], tf.int32)
                if 'elmo' in self.config.emb_class:
                    parsed['elmo_wordchr_ids'] = tf.reshape(tf.cast(parsed['elmo_wordchr_ids'], tf.int32), [-1, self.config.word_length])
                if self.build_output:
                    parsed['tags'] = tf.reshape(tf.cast(parsed['tags'], tf.int32), [-1, self.config.class_size])
            return parsed

        dataset = dataset.map(parser)
        if do_shuffle: dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        return dataset

    def __create_single_tf_example(self, bucket, ex_index, is_inference=False):
        """Create a single tf example
        """
        # create raw example
        example = {}
        if 'bert' in self.config.emb_class:
            bert_token_ids, bert_token_masks, bert_segment_ids, \
            bert_word_ids, bert_wordchr_ids, bert_pos_ids, bert_chk_ids, \
            bert_tags, bert_wordidx2tokenidx, bert_elmo_indices = \
                self.__create_bert_input(bucket, ex_index)
            example['word_ids'] = bert_word_ids                             # [bert_max_seq_length]
            example['wordchr_ids'] = bert_wordchr_ids                       # [bert_max_seq_length, word_length]
            example['pos_ids'] = bert_pos_ids                               # [bert_max_seq_length]
            example['chk_ids'] = bert_chk_ids                               # [bert_max_seq_length]
            example['bert_token_ids'] = bert_token_ids                      # [bert_max_seq_length]
            example['bert_token_masks'] = bert_token_masks                  # [bert_max_seq_length]
            example['bert_segment_ids'] = bert_segment_ids                  # [bert_max_seq_length]
            example['bert_wordidx2tokenidx'] = bert_wordidx2tokenidx        # [bert_max_seq_length]
            if 'elmo' in self.config.emb_class:
                example['bert_elmo_indices'] = bert_elmo_indices            # [bert_max_seq_length, 2]
                elmo_wordchr_ids = self.__create_elmo_wordchr_ids(bucket)
                example['elmo_wordchr_ids'] = elmo_wordchr_ids              # [bert_max_seq_length+2, word_length]
            if self.build_output:
                example['tags'] = bert_tags                                 # [bert_max_seq_length, class_size]
        else:
            word_ids = self.__create_word_ids(bucket)
            wordchr_ids = self.__create_wordchr_ids(bucket)
            example['word_ids'] = word_ids                                  # [max_sentence_length]
            example['wordchr_ids'] = wordchr_ids                            # [max_sentence_length, word_length]
            pos_ids = self.__create_pos_ids(bucket)
            chk_ids = self.__create_chk_ids(bucket)
            example['pos_ids'] = pos_ids                                    # [max_sentence_length]
            example['chk_ids'] = chk_ids                                    # [max_sentence_length]
            if self.config.emb_class == 'elmo':
                elmo_wordchr_ids = self.__create_elmo_wordchr_ids(bucket)
                example['elmo_wordchr_ids'] = elmo_wordchr_ids              # [max_sentence_length+2, word_length]
            if self.build_output:
                tags = self.__create_tags(bucket)
                example['tags'] = tags                                      # [max_sentence_length, class_size]

        if is_inference:
            for key, val in example.items():
                # expand dimension for batch size 1
                example[key] = [val]
            # no need to compute tf example for inference time
            return None, example

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        # create tf example(need to flat)
        features = collections.OrderedDict()
        if 'bert' in self.config.emb_class:
            features['word_ids'] = create_int_feature(example['word_ids'])
            t = np.reshape(example['wordchr_ids'], -1)
            features['wordchr_ids'] = create_int_feature(t)
            features['pos_ids'] = create_int_feature(example['pos_ids'])
            features['chk_ids'] = create_int_feature(example['chk_ids'])
            features['bert_token_ids'] = create_int_feature(example['bert_token_ids'])
            features['bert_token_masks'] = create_int_feature(example['bert_token_masks'])
            features['bert_segment_ids'] = create_int_feature(example['bert_segment_ids'])
            features['bert_wordidx2tokenidx'] = create_int_feature(example['bert_wordidx2tokenidx'])
            if 'elmo' in self.config.emb_class:
                t = np.reshape(example['bert_elmo_indices'], -1)
                features['bert_elmo_indices'] = create_int_feature(t)
                t = np.reshape(example['elmo_wordchr_ids'], -1)
                features['elmo_wordchr_ids'] = create_int_feature(t)
            if self.build_output:
                t = np.reshape(example['tags'], -1)
                features['tags'] = create_int_feature(t)
        else:
            features['word_ids'] = create_int_feature(example['word_ids'])
            t = np.reshape(example['wordchr_ids'], -1)
            features['wordchr_ids'] = create_int_feature(t)
            features['pos_ids'] = create_int_feature(example['pos_ids'])
            features['chk_ids'] = create_int_feature(example['chk_ids'])
            if 'elmo' in self.config.emb_class:
                t = np.reshape(example['elmo_wordchr_ids'], -1)
                features['elmo_wordchr_ids'] = create_int_feature(t)
            if self.build_output:
                t = np.reshape(example['tags'], -1)
                features['tags'] = create_int_feature(t)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        return tf_example, example

# -----------------------------------------------------------------------------
# convert input data to ids
# -----------------------------------------------------------------------------

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
               bert wordidx to tokenidx,
               bert for elmo indices
        """
        word_ids = self.__create_word_ids(bucket)
        wordchr_ids = self.__create_wordchr_ids(bucket)
        pos_ids = self.__create_pos_ids(bucket)
        chk_ids = self.__create_chk_ids(bucket)
        tags = self.__create_tags(bucket)

        bert_word_ids = []
        bert_wordchr_ids = []
        bert_pos_ids = []
        bert_chk_ids = []
        bert_tags = []

        bert_tokenizer = self.config.bert_tokenizer
        bert_max_seq_length = self.config.bert_max_seq_length
        ntokens = []
        bert_segment_ids = []
        bert_wordidx2tokenidx = []
        bert_elmo_indices = []
        ntokens_last = 0

        ntokens.append('[CLS]')
        bert_segment_ids.append(0)
        bert_elmo_indices.append([0,0])
        bert_word_ids.append(self.config.embvec.pad_wid) # 0
        pad_chr_ids = []
        for _ in range(self.config.word_length):
            pad_chr_ids.append(self.config.embvec.pad_cid) # 0
        bert_wordchr_ids.append(pad_chr_ids)
        bert_pos_ids.append(self.config.embvec.unk_pid) # 1, do not use pad_pid
        bert_chk_ids.append(self.config.embvec.unk_kid) # 1, unk_kid
        bert_tags.append(self.__tag_vec(self.config.embvec.oot_tag, self.config.class_size)) # 'O' tag
        ntokens_last += 1

        for i, line in enumerate(bucket):
            line = line.strip()
            tokens = line.split()
            assert (len(tokens) == 4)
            word = tokens[0]
            bert_tokens = bert_tokenizer.tokenize(word)
            for j, bert_token in enumerate(bert_tokens):
                ntokens.append(bert_token)
                bert_segment_ids.append(0)
                # extend bert_word_ids, bert_wordchr_ids, bert_pos_ids, bert_chk_ids, bert_tags
                bert_word_ids.append(word_ids[i])
                bert_wordchr_ids.append(wordchr_ids[i])
                bert_pos_ids.append(pos_ids[i])
                bert_chk_ids.append(chk_ids[i])
                if j == 0:
                    bert_tags.append(tags[i])
                    bert_wordidx2tokenidx.append(ntokens_last)
                else:
                    bert_tags.append(self.__tag_vec(self.config.embvec.xot_tag, self.config.class_size)) # 'X' tag
                bert_elmo_indices.append([ex_index, i])
                ntokens_last += 1
            if len(ntokens) == bert_max_seq_length - 1:
                tf.logging.debug('len(ntokens): %s' % str(len(ntokens)))
                break
        '''
        ntokens.append('[SEP]')
        bert_segment_ids.append(0)
        bert_word_ids.append(self.config.embvec.pad_wid) # 0
        bert_wordchr_ids.append(pad_chr_ids)
        bert_pos_ids.append(self.config.embvec.unk_pid) # 1, do not use pad_pid
        bert_chk_ids.append(self.config.embvec.unk_kid) # 1, unk_kid
        bert_tags.append(self.__tag_vec(self.config.embvec.oot_tag, self.config.class_size)) # 'O' tag
        ntokens_last += 1
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
        # padding for bert_word_ids, bert_wordchr_ids, bert_pos_ids, bert_chk_ids, bert_tags
        while len(bert_word_ids) < bert_max_seq_length:
            bert_word_ids.append(self.config.embvec.pad_wid)
            bert_wordchr_ids.append(pad_chr_ids)
            bert_pos_ids.append(self.config.embvec.pad_pid)
            bert_chk_ids.append(self.config.embvec.pad_kid)
            bert_tags.append(np.array([0] * self.config.class_size))
        assert len(bert_word_ids) == bert_max_seq_length
        assert len(bert_wordchr_ids) == bert_max_seq_length
        assert len(bert_pos_ids) == bert_max_seq_length
        assert len(bert_chk_ids) == bert_max_seq_length
        assert len(bert_tags) == bert_max_seq_length
        # padding for bert_wordidx2tokenidx (for FixedLenFeature())
        while len(bert_wordidx2tokenidx) < bert_max_seq_length:
            bert_wordidx2tokenidx.append(0)
        assert len(bert_wordidx2tokenidx) == bert_max_seq_length
        # padding for bert_elmo_indices
        while len(bert_elmo_indices) < bert_max_seq_length:
            bert_elmo_indices.append([0,0])
        assert len(bert_elmo_indices) == bert_max_seq_length

        if ex_index < 5:
            from bert import tokenization  
            tf.logging.debug('*** Example ***')
            tf.logging.debug('ntokens: %s' % ' '.join([tokenization.printable_text(x) for x in ntokens]))
            tf.logging.debug('bert_token_ids: %s' % ' '.join([str(x) for x in bert_token_ids]))
            tf.logging.debug('bert_token_masks: %s' % ' '.join([str(x) for x in bert_token_masks]))
            tf.logging.debug('bert_segment_ids: %s' % ' '.join([str(x) for x in bert_segment_ids]))
            tf.logging.debug('bert_word_ids: %s' % ' '.join([str(x) for x in bert_word_ids]))
            '''
            tf.logging.debug('bert_wordchr_ids: %s' % ' '.join([str(x) for x in bert_wordchr_ids]))
            tf.logging.debug('bert_pos_ids: %s' % ' '.join([str(x) for x in bert_pos_ids]))
            tf.logging.debug('bert_chk_ids: %s' % ' '.join([str(x) for x in bert_chk_ids]))
            tf.logging.debug('bert_tags: %s' % ' '.join([str(x) for x in bert_tags]))
            '''
            tf.logging.debug('bert_wordidx2tokenidx: %s' % ' '.join([str(x) for x in bert_wordidx2tokenidx]))
            tf.logging.debug('bert_elmo_indices: %s' % ' '.join([str(x) for x in bert_elmo_indices]))

        return bert_token_ids, bert_token_masks, bert_segment_ids, \
               bert_word_ids, bert_wordchr_ids, bert_pos_ids, bert_chk_ids, \
               bert_tags, bert_wordidx2tokenidx, bert_elmo_indices

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

    def __create_tags(self, bucket):
        """Create an output tag vector
        """
        tags  = []
        sentence_length = 0
        for line in bucket:
            line = line.strip()
            tokens = line.split()
            assert (len(tokens) == 4)
            sentence_length += 1
            tags.append(self.__tag_vec(tokens[3], self.config.class_size))   # tag one-hot
            if sentence_length == self.max_sentence_length: break
        # padding with 0s
        for _ in range(self.max_sentence_length - sentence_length):
            tags.append(np.array([0] * self.config.class_size))
        return tags

    def __tag_vec(self, tag, class_size):
        """Build one-hot vector for a tag
        """
        one_hot = np.zeros(class_size, dtype=np.int32)
        tid = self.config.embvec.get_tid(tag)
        one_hot[tid] = 1
        return one_hot

    @staticmethod
    def stat(file_name):
        temp_len = 0
        max_length = 0
        num_examples = 0
        for line in open(file_name):
            if line in ['\n', '\r\n']:
                if temp_len > max_length:
                    max_length = temp_len
                temp_len = 0
                num_examples += 1
            else:
                temp_len += 1
        return max_length, num_examples
