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

        # reference check
        default_graph = tf.get_default_graph()
        p_is_train = default_graph.get_tensor_by_name('is_train:0')
        p_sentence_length = default_graph.get_tensor_by_name('sentence_length:0')
        p_input_data_pos_ids = default_graph.get_tensor_by_name('input_data_pos_ids:0')
        p_wrd_embeddings_init = default_graph.get_tensor_by_name('wrd_embeddings_init:0')
        p_input_data_word_ids = default_graph.get_tensor_by_name('input_data_word_ids:0')
        p_input_data_wordchr_ids = default_graph.get_tensor_by_name('input_data_wordchr_ids:0')
        p_input_data_etcs = default_graph.get_tensor_by_name('input_data_etcs:0') 
        p_output_data = default_graph.get_tensor_by_name('output_data:0') # dummy
        t_logits = default_graph.get_tensor_by_name('logits:0')
        t_trans_params = default_graph.get_tensor_by_name('loss/trans_params:0')
        t_sentence_lengths = default_graph.get_tensor_by_name('sentence_lengths:0')
        print('is_train', p_is_train)
        print('sentence_length', p_sentence_length)
        print('input_data_pos_ids', p_input_data_pos_ids)
        print('wrd_embeddings_init', p_wrd_embeddings_init)
        print('input_data_word_ids', p_input_data_word_ids)
        print('input_data_wordchr_ids', p_input_data_wordchr_ids)
        print('input_data_etcs', p_input_data_etcs)
        print('output_data', p_output_data)
        print('logits', t_logits)
        print('trans_params', t_trans_params)
        print('sentence_lengths', t_sentence_lengths)

        # save
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, args.export)
        print('model exported')
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', type=str, help='path to saved model(ex, ../checkpoint/ner_model)', required=True)
    parser.add_argument('--export', type=str, help='path to exporting model(ex, exported/ner_model)', required=True)

    args = parser.parse_args()
    export(args)
