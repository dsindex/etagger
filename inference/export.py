from __future__ import print_function
import sys
import time
import argparse
import tensorflow as tf

def export(args):
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # restore
        meta_file = args.restore + '.meta'
        loader = tf.train.import_meta_graph(meta_file)
        loader.restore(sess, args.restore)
        print(tf.global_variables())
        print('model restored')
        # save
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, args.export)
        # reference check
        default_graph = tf.get_default_graph()
        is_train = default_graph.get_tensor_by_name('is_train:0')
        sentence_length = default_graph.get_tensor_by_name('sentence_length:0')
        input_data_pos_ids = default_graph.get_tensor_by_name('input_data_pos_ids:0')
        wrd_embeddings_init = default_graph.get_tensor_by_name('wrd_embeddings_init:0')
        input_data_word_ids = default_graph.get_tensor_by_name('input_data_word_ids:0')
        input_data_wordchr_ids = default_graph.get_tensor_by_name('input_data_wordchr_ids:0')
        input_data_etcs = default_graph.get_tensor_by_name('input_data_etcs:0') 
        logits = default_graph.get_tensor_by_name('logits:0')
        trans_params = default_graph.get_tensor_by_name('loss/trans_params:0') # FIXME can't be referred by 'trans_params:0'. why?
        print('is_train', is_train)
        print('sentence_length', sentence_length)
        print('input_data_pos_ids', input_data_pos_ids)
        print('wrd_embeddings_init', wrd_embeddings_init)
        print('input_data_word_ids', input_data_word_ids)
        print('input_data_wordchr_ids', input_data_wordchr_ids)
        print('input_data_etcs', input_data_etcs)
        print('logits', logits)
        print('trans_params', trans_params)
        print('model exported')
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', type=str, help='path to saved model(ex, ../checkpoint/ner_model)', required=True)
    parser.add_argument('--export', type=str, help='path to exporting model(ex, exported/ner_model)', required=True)

    args = parser.parse_args()
    export(args)
