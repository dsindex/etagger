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

def load_graph(frozen_graph_filename, prefix='prefix'):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            op_dict=None, 
            producer_op_list=None,
            name=prefix,
        )
        
    return graph

def inference(config, frozen_pb_path):
    """Inference for bucket
    """

    # load graph
    graph = load_graph(frozen_pb_path)
    for op in graph.get_operations():
        sys.stderr.write(op.name + '\n')

    # create session with graph
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    '''
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False,
                                  inter_op_parallelism_threads=1,
                                  intra_op_parallelism_threads=1)
    '''
    sess = tf.Session(graph=graph, config=session_conf)

    # mapping placeholders and tensors
    p_is_train = graph.get_tensor_by_name('prefix/is_train:0')
    p_sentence_length = graph.get_tensor_by_name('prefix/sentence_length:0')
    p_input_data_pos_ids = graph.get_tensor_by_name('prefix/input_data_pos_ids:0')
    p_wrd_embeddings_init = graph.get_tensor_by_name('prefix/wrd_embeddings_init:0')
    p_input_data_word_ids = graph.get_tensor_by_name('prefix/input_data_word_ids:0')
    p_input_data_wordchr_ids = graph.get_tensor_by_name('prefix/input_data_wordchr_ids:0')
    p_input_data_etcs = graph.get_tensor_by_name('prefix/input_data_etcs:0') 
    t_logits = graph.get_tensor_by_name('prefix/logits:0')
    t_trans_params = graph.get_tensor_by_name('prefix/loss/trans_params:0')
    t_sentence_lengths = graph.get_tensor_by_name('prefix/sentence_lengths:0')

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
    parser.add_argument('--frozen', type=str, help='path to frozen model(ex, ./exported/frozen_model.pb)', required=True)

    args = parser.parse_args()
    args.restore = None
    config = Config(args, arg_train=False, use_elmo=False, use_crf=True)
    inference(config, args.frozen)
