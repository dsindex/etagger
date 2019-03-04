from __future__ import print_function
import sys
import os
path = os.path.dirname(os.path.abspath(__file__)) + '/../..'
sys.path.append(path)
import time
import argparse
import tensorflow as tf
# for LSTMBlockFusedCell(), https://github.com/tensorflow/tensorflow/issues/23369
tf.contrib.rnn
import numpy as np

from embvec import EmbVec
from config import Config
from token_eval  import TokenEval
from chunk_eval  import ChunkEval
from input import Input

def load_frozen_graph(frozen_graph_filename, prefix='prefix'):
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
    graph = load_frozen_graph(frozen_pb_path)
    for op in graph.get_operations():
        sys.stderr.write(op.name + '\n')

    # create session with graph
    # if graph is optimized by tensorRT, then
    # from tensorflow.contrib import tensorrt as trt
    # gpu_ops = tf.GPUOptions(per_process_gpu_memory_fraction = 0.50)
    gpu_ops = tf.GPUOptions()
    '''
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_ops)
    '''
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False,
                                  gpu_options=gpu_ops,
                                  inter_op_parallelism_threads=1,
                                  intra_op_parallelism_threads=1)
    sess = tf.Session(graph=graph, config=session_conf)

    # mapping placeholders and tensors
    p_is_train = graph.get_tensor_by_name('prefix/is_train:0')
    p_sentence_length = graph.get_tensor_by_name('prefix/sentence_length:0')
    p_input_data_pos_ids = graph.get_tensor_by_name('prefix/input_data_pos_ids:0')
    p_input_data_chk_ids = graph.get_tensor_by_name('prefix/input_data_chk_ids:0')
    p_input_data_word_ids = graph.get_tensor_by_name('prefix/input_data_word_ids:0')
    p_input_data_wordchr_ids = graph.get_tensor_by_name('prefix/input_data_wordchr_ids:0')
    t_logits_indices = graph.get_tensor_by_name('prefix/logits_indices:0')
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
                         p_input_data_chk_ids: inp.sentence_chk_ids,
                         p_is_train: False,
                         p_sentence_length: inp.max_sentence_length}
            feed_dict[p_input_data_word_ids] = inp.sentence_word_ids
            feed_dict[p_input_data_wordchr_ids] = inp.sentence_wordchr_ids
            if config.emb_class == 'elmo':
                feed_dict[p_elmo_input_data_wordchr_ids] = inp.sentence_elmo_wordchr_ids
            if config.emb_class == 'bert':
                feed_dict[p_bert_input_data_token_ids] = inp.sentence_bert_token_ids
                feed_dict[p_bert_input_data_token_masks] = inp.sentence_bert_token_masks
                feed_dict[p_bert_input_data_segment_ids] = inp.sentence_bert_segment_ids
            logits_indices, sentence_lengths = sess.run([t_logits_indices, t_sentence_lengths], feed_dict=feed_dict)
            tags = inp.logit_indices_to_tags(logits_indices[0], sentence_lengths[0])
            for i in range(len(bucket)):
                if config.emb_class == 'bert':
                    j = inp.sentence_bert_wordidx2tokenidx[0][i]
                    out = bucket[i] + ' ' + tags[j]
                else:
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
                     p_input_data_chk_ids: inp.sentence_chk_ids,
                     p_is_train: False,
                     p_sentence_length: inp.max_sentence_length}
        feed_dict[p_input_data_word_ids] = inp.sentence_word_ids
        feed_dict[p_input_data_wordchr_ids] = inp.sentence_wordchr_ids
        if config.emb_class == 'elmo':
            feed_dict[p_elmo_input_data_wordchr_ids] = inp.sentence_elmo_wordchr_ids
        if config.emb_class == 'bert':
            feed_dict[p_bert_input_data_token_ids] = inp.sentence_bert_token_ids
            feed_dict[p_bert_input_data_token_masks] = inp.sentence_bert_token_masks
            feed_dict[p_bert_input_data_segment_ids] = inp.sentence_bert_segment_ids
        logits_indices, sentence_lengths = sess.run([t_logits_indices, t_sentence_lengths], feed_dict=feed_dict)
        tags = inp.logit_indices_to_tags(logits_indices[0], sentence_lengths[0])
        for i in range(len(bucket)):
            if config.emb_class == 'bert':
                j = inp.sentence_bert_wordidx2tokenidx[0][i]
                out = bucket[i] + ' ' + tags[j]
            else:
                out = bucket[i] + ' ' + tags[i]
            sys.stdout.write(out + '\n')
        sys.stdout.write('\n')
        duration_time = time.time() - start_time
        out = 'duration_time : ' + str(duration_time) + ' sec'
        tf.logging.info(out)
        num_buckets += 1
        total_duration_time += duration_time

    out = 'total_duration_time : ' + str(total_duration_time) + ' sec' + '\n'
    out += 'average processing time / bucket : ' + str(total_duration_time / num_buckets) + ' sec'
    tf.logging.info(out)

    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str, help='path to word embedding vector + vocab(.pkl)', required=True)
    parser.add_argument('--wrd_dim', type=int, help='dimension of word embedding vector', required=True)
    parser.add_argument('--word_length', type=int, default=15, help='max word length')
    parser.add_argument('--frozen_path', type=str, help='path to frozen model(ex, ./exported/ner_frozen.pb)', required=True)

    args = parser.parse_args()
    tf.logging.set_verbosity(tf.logging.INFO)

    args.restore = None
    config = Config(args, is_training=False, emb_class='glove', use_crf=True)
    inference(config, args.frozen_path)
