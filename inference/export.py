from __future__ import print_function
import sys
import time
import argparse
import tensorflow as tf
# for LSTMBlockFusedCell(), https://github.com/tensorflow/tensorflow/issues/23369
tf.contrib.rnn

def export(args):
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # restore meta graph
        meta_file = args.restore + '.meta'
        loader = tf.train.import_meta_graph(meta_file, clear_devices=True)
        # mapping placeholders and tensors
        graph = tf.get_default_graph()
        p_is_train = graph.get_tensor_by_name('is_train:0')
        p_sentence_length = graph.get_tensor_by_name('sentence_length:0')
        p_input_data_pos_ids = graph.get_tensor_by_name('input_data_pos_ids:0')
        p_input_data_chk_ids = graph.get_tensor_by_name('input_data_chk_ids:0')
        p_input_data_word_ids = graph.get_tensor_by_name('input_data_word_ids:0')
        p_input_data_wordchr_ids = graph.get_tensor_by_name('input_data_wordchr_ids:0')
        t_logits = graph.get_tensor_by_name('logits:0')
        t_trans_params = graph.get_tensor_by_name('loss/trans_params:0')
        t_sentence_lengths = graph.get_tensor_by_name('sentence_lengths:0')
        print('is_train', p_is_train)
        print('sentence_length', p_sentence_length)
        print('input_data_pos_ids', p_input_data_pos_ids)
        print('input_data_chk_ids', p_input_data_chk_ids)
        print('input_data_word_ids', p_input_data_word_ids)
        print('input_data_wordchr_ids', p_input_data_wordchr_ids)
        print('logits', t_logits)
        print('trans_params', t_trans_params)
        print('sentence_lengths', t_sentence_lengths)
        # restore actual values
        loader.restore(sess, args.restore)
        print(tf.global_variables())
        print(tf.trainable_variables())
        print('model restored')

        # save
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, args.export)
        tf.train.write_graph(sess.graph, args.export_pb, "graph.pb", as_text=False)
        tf.train.write_graph(sess.graph, args.export_pb, "graph.pb_txt", as_text=True)
        print('model exported')
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', type=str, help='path to saved model(ex, ../checkpoint/ner_model)', required=True)
    parser.add_argument('--export', type=str, help='path to exporting model(ex, exported/ner_model)', required=True)
    parser.add_argument('--export-pb', type=str, help='path to exporting graph proto(ex, exported)', required=True)

    args = parser.parse_args()
    export(args)
