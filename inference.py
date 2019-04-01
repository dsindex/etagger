from __future__ import print_function
import sys
import time
import argparse
import tensorflow as tf
import numpy as np
from embvec import EmbVec
from config import Config
from model import Model
from token_eval  import TokenEval
from chunk_eval  import ChunkEval
from input import Input

def inference_bucket(config):
    """Inference for bucket.
    """

    # Create model
    model = Model(config)

    # Restore model
    session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    '''
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False,
                                  inter_op_parallelism_threads=1,
                                  intra_op_parallelism_threads=1)
    '''
    sess = tf.compat.v1.Session(config=session_conf)
    feed_dict = {model.wrd_embeddings_init: config.embvec.wrd_embeddings}
    sess.run(tf.compat.v1.global_variables_initializer(), feed_dict=feed_dict)
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, config.restore)
    sys.stderr.write('model restored' +'\n')
    '''
    print(tf.global_variables())
    print(tf.trainable_variables())
    '''
    num_buckets = 0
    total_duration_time = 0.0
    bucket = []
    while 1:
        try: line = sys.stdin.readline()
        except KeyboardInterrupt: break
        if not line: break
        line = line.strip()
        if not line and len(bucket) >= 1:
            start_time = time.time()
            # Build input data
            inp = Input(bucket, config, build_output=False)
            feed_dict = {model.input_data_pos_ids: inp.example['pos_ids'],
                         model.input_data_chk_ids: inp.example['chk_ids'],
                         model.is_train: False,
                         model.sentence_length: inp.max_sentence_length}
            feed_dict[model.input_data_word_ids] = inp.example['word_ids']
            feed_dict[model.input_data_wordchr_ids] = inp.example['wordchr_ids']
            if 'elmo' in config.emb_class:
                feed_dict[model.elmo_input_data_wordchr_ids] = inp.example['elmo_wordchr_ids']
            if 'bert' in config.emb_class:
                feed_dict[model.bert_input_data_token_ids] = inp.example['bert_token_ids']
                feed_dict[model.bert_input_data_token_masks] = inp.example['bert_token_masks']
                feed_dict[model.bert_input_data_segment_ids] = inp.example['bert_segment_ids']
                if 'elmo' in config.emb_class:
                    feed_dict[model.bert_input_data_elmo_indices] = inp.example['bert_elmo_indices']
            logits_indices, sentence_lengths = sess.run([model.logits_indices, model.sentence_lengths], feed_dict=feed_dict)
            tags = config.logit_indices_to_tags(logits_indices[0], sentence_lengths[0])
            for i in range(len(bucket)):
                if 'bert' in config.emb_class:
                    j = inp.example['bert_wordidx2tokenidx'][0][i]
                    out = bucket[i] + ' ' + tags[j]
                else:
                    out = bucket[i] + ' ' + tags[i]
                sys.stdout.write(out + '\n')
            sys.stdout.write('\n')
            bucket = []
            duration_time = time.time() - start_time
            out = 'duration_time : ' + str(duration_time) + ' sec'
            tf.compat.v1.logging.info(out)
            num_buckets += 1
            total_duration_time += duration_time
        if line : bucket.append(line)
    if len(bucket) != 0:
        start_time = time.time()
        # Build input data
        inp = Input(bucket, config, build_output=False)
        feed_dict = {model.input_data_pos_ids: inp.example['pos_ids'],
                     model.input_data_chk_ids: inp.example['chk_ids'],
                     model.is_train: False,
                     model.sentence_length: inp.max_sentence_length}
        feed_dict[model.input_data_word_ids] = inp.example['word_ids']
        feed_dict[model.input_data_wordchr_ids] = inp.example['wordchr_ids']
        if 'elmo' in config.emb_class:
            feed_dict[model.elmo_input_data_wordchr_ids] = inp.example['elmo_wordchr_ids']
        if 'bert' in config.emb_class:
            feed_dict[model.bert_input_data_token_ids] = inp.example['bert_token_ids']
            feed_dict[model.bert_input_data_token_masks] = inp.example['bert_token_masks']
            feed_dict[model.bert_input_data_segment_ids] = inp.example['bert_segment_ids']
            if 'elmo' in config.emb_class:
                feed_dict[model.bert_input_data_elmo_indices] = inp.example['bert_elmo_indices']
        logits_indices, sentence_lengths = sess.run([model.logits_indices, model.sentence_lengths], feed_dict=feed_dict)
        tags = config.logit_indices_to_tags(logits_indices[0], sentence_lengths[0])
        for i in range(len(bucket)):
            if 'bert' in config.emb_class:
                j = inp.example['bert_wordidx2tokenidx'][0][i]
                out = bucket[i] + ' ' + tags[j]
            else:
                out = bucket[i] + ' ' + tags[i]
            sys.stdout.write(out + '\n')
        sys.stdout.write('\n')
        duration_time = time.time() - start_time
        out = 'duration_time : ' + str(duration_time) + ' sec'
        tf.compat.v1.logging.info(out)
        num_buckets += 1
        total_duration_time += duration_time

    out = 'total_duration_time : ' + str(total_duration_time) + ' sec' + '\n'
    out += 'average processing time / bucket : ' + str(total_duration_time / num_buckets) + ' sec'
    tf.compat.v1.logging.info(out)

    sess.close()

def inference_line(config):
    """Inference for raw string.
    """
    def get_entity(doc, begin, end):
        for ent in doc.ents:
            # check included
            if ent.start_char <= begin and end <= ent.end_char:
                if ent.start_char == begin: return 'B-' + ent.label_
                else: return 'I-' + ent.label_
        return 'O'
     
    def build_bucket(nlp, line):
        bucket = []
        doc = nlp(line)
        for token in doc:
            begin = token.idx
            end   = begin + len(token.text) - 1
            temp = []
            '''
            print(token.i, token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                  token.shape_, token.is_alpha, token.is_stop, begin, end)
            '''
            temp.append(token.text)
            temp.append(token.tag_)
            temp.append('O')     # no chunking info
            entity = get_entity(doc, begin, end)
            temp.append(entity)  # entity by spacy
            temp = ' '.join(temp)
            bucket.append(temp)
        return bucket

    import spacy
    nlp = spacy.load('en')

    # Create model
    model = Model(config)

    # Restore model
    session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.compat.v1.Session(config=session_conf)
    feed_dict = {}
    feed_dict = {model.wrd_embeddings_init: config.embvec.wrd_embeddings}
    sess.run(tf.compat.v1.global_variables_initializer(), feed_dict=feed_dict)
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, config.restore)
    tf.compat.v1.logging.info('model restored' +'\n')

    while 1:
        try: line = sys.stdin.readline()
        except KeyboardInterrupt: break
        if not line: break
        line = line.strip()
        if not line: continue
        # Create bucket
        try: bucket = build_bucket(nlp, line)
        except Exception as e:
            sys.stderr.write(str(e) +'\n')
            continue
        # Build input data
        inp = Input(bucket, config, build_output=False)
        feed_dict = {model.input_data_pos_ids: inp.example['pos_ids'],
                     model.input_data_chk_ids: inp.example['chk_ids'],
                     model.is_train: False,
                     model.sentence_length: inp.max_sentence_length}
        feed_dict[model.input_data_word_ids] = inp.example['word_ids']
        feed_dict[model.input_data_wordchr_ids] = inp.example['wordchr_ids']
        if 'elmo' in config.emb_class:
            feed_dict[model.elmo_input_data_wordchr_ids] = inp.example['elmo_wordchr_ids']
        if 'bert' in config.emb_class:
            feed_dict[model.bert_input_data_token_ids] = inp.example['bert_token_ids']
            feed_dict[model.bert_input_data_token_masks] = inp.example['bert_token_masks']
            feed_dict[model.bert_input_data_segment_ids] = inp.example['bert_segment_ids']
            if 'elmo' in config.emb_class:
                feed_dict[model.bert_input_data_elmo_indices] = inp.example['bert_elmo_indices']
        logits_indices, sentence_lengths = sess.run([model.logits_indices, model.sentence_lengths], feed_dict=feed_dict)
        tags = config.logit_indices_to_tags(logits_indices[0], sentence_lengths[0])
        for i in range(len(bucket)):
            if 'bert' in config.emb_class:
                j = inp.example['bert_wordidx2tokenidx'][0][i]
                out = bucket[i] + ' ' + tags[j]
            else:
                out = bucket[i] + ' ' + tags[i]
            sys.stdout.write(out + '\n')
        sys.stdout.write('\n')

    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str, help='path to word embedding vector + vocab(.pkl)', required=True)
    parser.add_argument('--wrd_dim', type=int, help='dimension of word embedding vector', required=True)
    parser.add_argument('--word_length', type=int, default=15, help='max word length')
    parser.add_argument('--restore', type=str, help='path to saved model(ex, ./checkpoint/ner_model)', required=True)
    parser.add_argument('--mode', type=str, default='bulk', help='bulk, bucket, line')

    args = parser.parse_args()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    config = Config(args, is_training=False, emb_class='glove', use_crf=True)
    if args.mode == 'bucket': inference_bucket(config)
    if args.mode == 'line':   inference_line(config)
