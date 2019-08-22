from __future__ import print_function
import sys
import os
path = os.path.dirname(os.path.abspath(__file__)) + '/../..'
sys.path.append(path)
import time
import argparse
import tensorflow as tf
import numpy as np
# for LSTMBlockFusedCell(), https://github.com/tensorflow/tensorflow/issues/23369
tf.contrib.rnn
# for QRNN
try: import qrnn
except: sys.stderr.write('import qrnn, failed\n')

from embvec import EmbVec
from config import Config
from input import Input
import feed

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
                                  inter_op_parallelism_threads=0,
                                  intra_op_parallelism_threads=0)
    sess = tf.Session(graph=graph, config=session_conf)

    # mapping output/input tensors for bert
    if 'bert' in config.emb_class:
        t_bert_embeddings_subgraph = graph.get_tensor_by_name('prefix/bert_embeddings_subgraph:0')
        p_bert_embeddings = graph.get_tensor_by_name('prefix/bert_embeddings:0')
    # mapping output tensors
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
            inp, feed_dict = feed.build_input_feed_dict_with_graph(graph, config, bucket, Input)
            if 'bert' in config.emb_class:
                # compute bert embedding at runtime
                bert_embeddings = sess.run([t_bert_embeddings_subgraph], feed_dict=feed_dict)
                # update feed_dict
                feed_dict[p_bert_embeddings] = feed.align_bert_embeddings(config, bert_embeddings, inp.example['bert_wordidx2tokenidx'], -1)
            logits_indices, sentence_lengths = sess.run([t_logits_indices, t_sentence_lengths], feed_dict=feed_dict)
            tags = config.logit_indices_to_tags(logits_indices[0], sentence_lengths[0])
            for i in range(len(bucket)):
                out = bucket[i] + ' ' + tags[i]
                sys.stdout.write(out + '\n')
            sys.stdout.write('\n')
            bucket = []
            duration_time = time.time() - start_time
            out = 'duration_time : ' + str(duration_time) + ' sec'
            sys.stderr.write(out + '\n')
            num_buckets += 1
            if num_buckets != 1: # first one may takes longer time, so ignore in computing duration.
                total_duration_time += duration_time
        if line : bucket.append(line)
    if len(bucket) != 0:
        start_time = time.time()
        inp, feed_dict = feed.build_input_feed_dict_with_graph(graph, config, bucket, Input)
        if 'bert' in config.emb_class:
            # compute bert embedding at runtime
            bert_embeddings = sess.run([t_bert_embeddings_subgraph], feed_dict=feed_dict)
            # update feed_dict
            feed_dict[p_bert_embeddings] = feed.align_bert_embeddings(config, bert_embeddings, inp.example['bert_wordidx2tokenidx'], -1)
        logits_indices, sentence_lengths = sess.run([t_logits_indices, t_sentence_lengths], feed_dict=feed_dict)
        tags = config.logit_indices_to_tags(logits_indices[0], sentence_lengths[0])
        for i in range(len(bucket)):
            out = bucket[i] + ' ' + tags[i]
            sys.stdout.write(out + '\n')
        sys.stdout.write('\n')
        duration_time = time.time() - start_time
        out = 'duration_time : ' + str(duration_time) + ' sec'
        tf.logging.info(out)
        num_buckets += 1
        total_duration_time += duration_time

    out = 'total_duration_time : ' + str(total_duration_time) + ' sec' + '\n'
    out += 'average processing time / bucket : ' + str(total_duration_time / (num_buckets-1)) + ' sec'
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
