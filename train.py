from __future__ import print_function
import tensorflow as tf
import numpy as np
from config import Config
from model import Model
from token_eval  import Eval
from input import *
import os
import sys
import argparse

def do_train(model, config, train_data, dev_data, test_data):
    learning_rate_init=0.001  # initial
    learning_rate_final=0.0001 # final
    learning_rate=learning_rate_init
    intermid_epoch = 20       # after this epoch, change learning rate
    maximum = 0
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if config.restore is not None:
            saver.restore(sess, config.restore)
            print('model restored')
        # summary for loss, accuracy
        loss_summary = tf.summary.scalar('loss', model.loss)
        acc_summary = tf.summary.scalar('accuracy', model.accuracy)
        # train summary
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(config.summary_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        # dev summary
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(config.summary_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        # training steps
        for e in range(config.epoch):
            idx = 0
            for ptr in range(0, len(train_data.sentence_word_ids), config.batch_size):
                print('%s-th batch in %s(size of train_inp)' % (idx, len(train_data.sentence_word_ids)))
                feed_dict={model.input_data_word_ids: train_data.sentence_word_ids[ptr:ptr + config.batch_size],
                           model.input_data_wordchr_ids: train_data.sentence_wordchr_ids[ptr:ptr + config.batch_size],
                           model.input_data_pos_ids: train_data.sentence_pos_ids[ptr:ptr + config.batch_size],
                           model.input_data_etc: train_data.sentence_etc[ptr:ptr + config.batch_size],
                           model.output_data: train_data.sentence_tag[ptr:ptr + config.batch_size],
                           model.learning_rate:learning_rate}
                step, train_summaries, _, train_loss, train_accuracy, rnn_output, attended_output = \
                           sess.run([model.global_step, train_summary_op, model.train_op, model.loss, model.accuracy, model.rnn_output, model.attended_output], feed_dict=feed_dict)
                print('step: %d, train loss: %s, train accuracy: %s' % (step, train_loss, train_accuracy))
                train_summary_writer.add_summary(train_summaries, step)
                idx += 1
            # learning rate warmup
            if e > intermid_epoch: learning_rate=learning_rate_final
            if e % 10 == 0:
                save_path = saver.save(sess, config.checkpoint_dir + '/' + 'model.ckpt')
                print('model saved in file: %s' % save_path)
            feed_dict={model.input_data_word_ids: dev_data.sentence_word_ids,
                       model.input_data_wordchr_ids: dev_data.sentence_wordchr_ids,
                       model.input_data_pos_ids: dev_data.sentence_pos_ids,
                       model.input_data_etc: dev_data.sentence_etc,
                       model.output_data: dev_data.sentence_tag}
            step, dev_summaries, pred, length, dev_loss, dev_accuracy = sess.run([model.global_step, dev_summary_op, model.prediction, model.length, model.loss, model.accuracy], feed_dict=feed_dict)
            print('epoch: %d, step: %d, dev loss: %s, dev accuracy: %s' % (e, step, dev_loss, dev_accuracy))
            dev_summary_writer.add_summary(dev_summaries, step)
            print('dev precision, recall, f1:')
            m = Eval.compute_f1(config.class_size, pred, dev_data.sentence_tag, length)
            if m > maximum:
                maximum = m
                save_path = saver.save(sess, config.checkpoint_dir + '/' + 'model_max.ckpt')
                print('max model saved in file: %s' % save_path)
                feed_dict={model.input_data_word_ids: test_data.sentence_word_ids,
                           model.input_data_wordchr_ids: test_data.sentence_wordchr_ids,
                           model.input_data_pos_ids: test_data.sentence_pos_ids,
                           model.input_data_etc: test_data.sentence_etc,
                           model.output_data: test_data.sentence_tag}
                step, pred, length, test_loss, test_accuracy = sess.run([model.global_step, model.prediction, model.length, model.loss, model.accuracy], feed_dict=feed_dict)
                print('epoch: %d, step: %d, test loss: %s, test accuracy: %s' % (e, step, test_loss, test_accuracy))
                print('test precision, recall, f1:')
                Eval.compute_f1(config.class_size, pred, test_data.sentence_tag, length)

def train(config):
    # Build input data
    train_file = 'data/train.txt'
    dev_file = 'data/dev.txt'
    test_file = 'data/test.txt'
    train_data = Input(train_file, config)
    print('max_sentence_length = %d' % train_data.max_sentence_length)
    dev_data = Input(dev_file, config)
    print('max_sentence_length = %d' % dev_data.max_sentence_length)
    test_data = Input(test_file, config)
    print('max_sentence_length = %d' % test_data.max_sentence_length)
    print('loading input data ... done')

    # Create model
    model = Model(config)

    # Training
    do_train(model, config, train_data, dev_data, test_data)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str, help='path to word embedding vector(.pkl)', required=True)
    parser.add_argument('--wrd_dim', type=int, help='dimension of word embedding vector', required=True)
    parser.add_argument('--sentence_length', type=int, help='max sentence length', required=True)
    parser.add_argument('--word_length', type=int, default=15, help='max word length')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='dir path to save model(ex, ./checkpoint)')
    parser.add_argument('--restore', type=str, default=None, help='path to saved model(ex, ./checkpoint/model.ckpt)')
    parser.add_argument('--summary_dir', type=str, default='./runs', help='path to save summary(ex, ./runs)')

    args = parser.parse_args()
    config = Config(args, is_train=True)
    train(config)
