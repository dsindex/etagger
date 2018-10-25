from __future__ import print_function
import sys
import os
path = os.path.dirname(os.path.abspath(__file__)) + '/../..'
sys.path.append(path)
import time
import argparse
import tensorflow as tf
import numpy as np

from embvec import EmbVec
from config import Config
from token_eval  import TokenEval
from chunk_eval  import ChunkEval
from viterbi import viterbi_decode
from input import Input

def inference(config):
    """Inference for bucket
    """

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    '''
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False,
                                  inter_op_parallelism_threads=1,
                                  intra_op_parallelism_threads=1)
    '''
    sess = tf.Session(config=session_conf)
    # restore meta graph
    meta_file = config.restore + '.meta'
    loader = tf.train.import_meta_graph(meta_file)
    # mapping placeholders and tensors
    graph = tf.get_default_graph()
    p_is_train = graph.get_tensor_by_name('is_train:0')
    p_sentence_length = graph.get_tensor_by_name('sentence_length:0')
    p_input_data_pos_ids = graph.get_tensor_by_name('input_data_pos_ids:0')
    p_wrd_embeddings_init = graph.get_tensor_by_name('wrd_embeddings_init:0')
    p_input_data_word_ids = graph.get_tensor_by_name('input_data_word_ids:0')
    p_input_data_wordchr_ids = graph.get_tensor_by_name('input_data_wordchr_ids:0')
    p_input_data_etcs = graph.get_tensor_by_name('input_data_etcs:0') 
    t_logits = graph.get_tensor_by_name('logits:0')
    t_trans_params = graph.get_tensor_by_name('loss/trans_params:0')
    t_sentence_lengths = graph.get_tensor_by_name('sentence_lengths:0')
    # run global_variables_initializer() with feed_dict first
    feed_dict = {}
    if not config.use_elmo: feed_dict = {p_wrd_embeddings_init: config.embvec.wrd_embeddings}
    sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
    # restore actual values
    loader.restore(sess, config.restore)
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
            inp = Input(bucket, config, build_output=False)
            feed_dict = {p_input_data_pos_ids: inp.sentence_pos_ids,
                         p_input_data_etcs: inp.sentence_etcs,
                         p_is_train: False,
                         p_sentence_length: inp.max_sentence_length}
            if config.use_elmo:
                feed_dict[p_elmo_input_data_wordchr_ids] = inp.sentence_elmo_wordchr_ids
            else:
                feed_dict[p_input_data_word_ids] = inp.sentence_word_ids
                feed_dict[p_input_data_wordchr_ids] = inp.sentence_wordchr_ids
            logits, trans_params, sentence_lengths = sess.run([t_logits, t_trans_params, t_sentence_lengths], \
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
        inp = Input(bucket, config, build_output=False)
        feed_dict = {p_input_data_pos_ids: inp.sentence_pos_ids,
                     p_input_data_etcs: inp.sentence_etcs,
                     p_is_train: False,
                     p_sentence_length: inp.max_sentence_length}
        if config.use_elmo:
            feed_dict[p_elmo_input_data_wordchr_ids] = inp.sentence_elmo_wordchr_ids
        else:
            feed_dict[p_input_data_word_ids] = inp.sentence_word_ids
            feed_dict[p_input_data_wordchr_ids] = inp.sentence_wordchr_ids
        logits, trans_params, sentence_lengths = sess.run([t_logits, t_trans_params, t_sentence_lengths], \
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
    parser.add_argument('--restore', type=str, help='path to saved model(ex, ./exported/ner_model)', required=True)

    args = parser.parse_args()
    config = Config(args, arg_train=False, use_elmo=False, use_crf=True)
    inference(config)
