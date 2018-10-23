from __future__ import print_function
import sys
import time
import argparse
import tensorflow as tf
import numpy as np
sys.path.append('../..')

from embvec import EmbVec
from config import Config
from model import Model
from token_eval  import TokenEval
from chunk_eval  import ChunkEval
from viterbi import viterbi_decode
from input import Input

def export_model(config):
    """Inference for bucket
    """

    # Create model
    model = Model(config)

    # Restore model
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    '''
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False,
                                  inter_op_parallelism_threads=1,
                                  intra_op_parallelism_threads=1)
    '''
    sess = tf.Session(config=session_conf)
    feed_dict = {}
    if not config.use_elmo: feed_dict = {model.wrd_embeddings_init: config.embvec.wrd_embeddings}
    sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
    saver = tf.train.Saver()
    saver.restore(sess, config.restore)
    sys.stderr.write('model restored' +'\n')

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
            inp = Input(bucket, config)
            feed_dict = {model.input_data_pos_ids: inp.sentence_pos_ids,
                         model.input_data_etcs: inp.sentence_etcs,
                         model.output_data: inp.sentence_tags,
                         model.is_train: False,
                         model.sentence_length: inp.max_sentence_length}
            if config.use_elmo:
                feed_dict[model.elmo_input_data_wordchr_ids] = inp.sentence_elmo_wordchr_ids
            else:
                feed_dict[model.input_data_word_ids] = inp.sentence_word_ids
                feed_dict[model.input_data_wordchr_ids] = inp.sentence_wordchr_ids
            logits, trans_params, sentence_lengths, loss = \
                         sess.run([model.logits, model.trans_params, \
                                   model.sentence_lengths, model.loss], \
                                   feed_dict=feed_dict)
            if config.use_crf:
                viterbi_sequences = viterbi_decode(logits, trans_params, sentence_lengths)
                tags = inp.logit_indices_to_tags(viterbi_sequences[0], sentence_lengths[0])
            else:
                tags = inp.logit_to_tags(logits[0], sentence_lengths[0])
            for i in range(len(bucket)):
                out = bucket[i] + ' ' + tags[i]
                sys.stdout.write(out + '\n')
            sys.stdout.write('\n')
            bucket = []
            duration_time = time.time() - start_time
            out = 'duration_time : ' + str(duration_time) + ' sec'
            sys.stderr.write(out + '\n')
            num_buckets += 1
            total_duration_time += duration_time
        if line : bucket.append(line)
    if len(bucket) != 0:
        start_time = time.time()
        # Build input data
        inp = Input(bucket, config)
        feed_dict = {model.input_data_pos_ids: inp.sentence_pos_ids,
                     model.input_data_etcs: inp.sentence_etcs,
                     model.output_data: inp.sentence_tags,
                     model.is_train: False,
                     model.sentence_length: inp.max_sentence_length}
        if config.use_elmo:
            feed_dict[model.elmo_input_data_wordchr_ids] = inp.sentence_elmo_wordchr_ids
        else:
            feed_dict[model.input_data_word_ids] = inp.sentence_word_ids
            feed_dict[model.input_data_wordchr_ids] = inp.sentence_wordchr_ids
        logits, trans_params, sentence_lengths, loss = \
                     sess.run([model.logits, model.trans_params, \
                               model.sentence_lengths, model.loss], \
                               feed_dict=feed_dict)
        if config.use_crf:
            viterbi_sequences = viterbi_decode(logits, trans_params, sentence_lengths)
            tags = inp.logit_indices_to_tags(viterbi_sequences[0], sentence_lengths[0])
        else:
            tags = inp.logit_to_tags(logits[0], sentence_lengths[0])
        for i in range(len(bucket)):
            out = bucket[i] + ' ' + tags[i]
            sys.stdout.write(out + '\n')
        sys.stdout.write('\n')
        duration_time = time.time() - start_time
        out = 'duration_time : ' + str(duration_time) + ' sec'
        sys.stderr.write(out + '\n')
        num_buckets += 1
        total_duration_time += duration_time

    out = 'total_duration_time : ' + str(total_duration_time) + ' sec' + '\n'
    out += 'average processing time / bucket : ' + str(total_duration_time / num_buckets) + ' sec'
    sys.stderr.write(out + '\n')

    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str, help='path to word embedding vector(.pkl)', required=True)
    parser.add_argument('--wrd_dim', type=int, help='dimension of word embedding vector', required=True)
    parser.add_argument('--word_length', type=int, default=15, help='max word length')
    parser.add_argument('--restore', type=str, help='path to saved model(ex, ../checkpoint/ner_model)', required=True)

    args = parser.parse_args()
    config = Config(args, is_train=False, use_elmo=False, use_crf=True)
    export_model(config)
