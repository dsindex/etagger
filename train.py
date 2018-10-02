from __future__ import print_function
import tensorflow as tf
import numpy as np
from embvec import EmbVec
from config import Config
from model import Model
from input import Input
from token_eval  import TokenEval
from chunk_eval  import ChunkEval
from viterbi import viterbi_decode
from general_utils import Progbar
from early_stopping import EarlyStopping
import os
import sys
import random
import argparse

def do_train(model, config, train_data, dev_data, test_data):
    early_stopping = EarlyStopping(patience=10, measure='f1', verbose=1)
    maximum = 0
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        sess.run(tf.global_variables_initializer(), feed_dict={model.wrd_embeddings_init: config.embvec.wrd_embeddings}) # feed large embedding data
        saver = tf.train.Saver()
        if config.restore is not None:
            saver.restore(sess, config.restore)
            print('model restored')
        # summary setting
        loss_summary = tf.summary.scalar('loss', model.loss)
        acc_summary = tf.summary.scalar('accuracy', model.accuracy)
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(config.summary_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(config.summary_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        # training steps
        for e in range(config.epoch):
            # run epoch
            idx = 0
            nbatches = (len(train_data.sentence_word_ids) + config.batch_size - 1) // config.batch_size
            prog = Progbar(target=nbatches)
            for ptr in range(0, len(train_data.sentence_word_ids), config.batch_size):
                feed_dict={model.input_data_word_ids: train_data.sentence_word_ids[ptr:ptr + config.batch_size],
                           model.input_data_wordchr_ids: train_data.sentence_wordchr_ids[ptr:ptr + config.batch_size],
                           model.input_data_pos_ids: train_data.sentence_pos_ids[ptr:ptr + config.batch_size],
                           model.input_data_etcs: train_data.sentence_etcs[ptr:ptr + config.batch_size],
                           model.output_data: train_data.sentence_tags[ptr:ptr + config.batch_size],
                           model.is_train: True}
                step, train_summaries, _, train_loss, train_accuracy, learning_rate = \
                           sess.run([model.global_step, train_summary_op, model.train_op, \
                                     model.loss, model.accuracy, model.learning_rate], feed_dict=feed_dict)
                train_summary_writer.add_summary(train_summaries, step)
                prog.update(idx + 1, [('step', step), ('train loss', train_loss), ('train accuracy', train_accuracy), ('lr', learning_rate)])
                idx += 1
            # evaluate on dev data
            feed_dict={model.input_data_word_ids: dev_data.sentence_word_ids,
                       model.input_data_wordchr_ids: dev_data.sentence_wordchr_ids,
                       model.input_data_pos_ids: dev_data.sentence_pos_ids,
                       model.input_data_etcs: dev_data.sentence_etcs,
                       model.output_data: dev_data.sentence_tags,
                       model.is_train: False}
            step, dev_summaries, logits, logits_indices, trans_params, output_data_indices, sentence_lengths, dev_loss, dev_accuracy = \
                       sess.run([model.global_step, dev_summary_op, model.logits, model.logits_indices, \
                                 model.trans_params, model.output_data_indices, model.sentence_lengths, model.loss, model.accuracy], \
                                 feed_dict=feed_dict)
            print('epoch: %d / %d, step: %d, dev loss: %s, dev accuracy: %s' % (e, config.epoch, step, dev_loss, dev_accuracy))
            dev_summary_writer.add_summary(dev_summaries, step)
            print('dev precision, recall, f1(token): ')
            token_f1 = TokenEval.compute_f1(config.class_size, logits, dev_data.sentence_tags, sentence_lengths)
            '''
            # early stopping
            if early_stopping.validate(token_f1, measure='f1'): break
            '''
            '''
            if config.use_crf:
                viterbi_sequences = viterbi_decode(logits, trans_params, sentence_lengths)
                tag_preds = dev_data.logits_indices_to_tags_seq(viterbi_sequences, sentence_lengths)
            else:
                tag_preds = dev_data.logits_indices_to_tags_seq(logits_indices, sentence_lengths)
            tag_corrects = dev_data.logits_indices_to_tags_seq(output_data_indices, sentence_lengths)
            dev_prec, dev_rec, dev_f1 = ChunkEval.compute_f1(tag_preds, tag_corrects)
            print('dev precision, recall, f1(chunk): ', dev_prec, dev_rec, dev_f1)
            chunk_f1 = dev_f1
            '''
            # save best model
            '''
            m = chunk_f1 # slightly lower than token-based f1 for test
            '''
            m = token_f1
            if m > maximum:
                print('new best f1 score!')
                maximum = m
                save_path = saver.save(sess, config.checkpoint_dir + '/' + 'model_max.ckpt')
                print('max model saved in file: %s' % save_path)
                feed_dict={model.input_data_word_ids: test_data.sentence_word_ids,
                           model.input_data_wordchr_ids: test_data.sentence_wordchr_ids,
                           model.input_data_pos_ids: test_data.sentence_pos_ids,
                           model.input_data_etcs: test_data.sentence_etcs,
                           model.output_data: test_data.sentence_tags,
                           model.is_train: False}
                step, logits, logits_indices, trans_params, output_data_indices, sentence_lengths, test_loss, test_accuracy = \
                           sess.run([model.global_step, model.logits, model.logits_indices, \
                                     model.trans_params, model.output_data_indices, model.sentence_lengths, model.loss, model.accuracy], \
                                     feed_dict=feed_dict)
                print('epoch: %d / %d, step: %d, test loss: %s, test accuracy: %s' % (e, config.epoch, step, test_loss, test_accuracy))
                print('test precision, recall, f1(token): ')
                TokenEval.compute_f1(config.class_size, logits, test_data.sentence_tags, sentence_lengths)
                if config.use_crf:
                    viterbi_sequences = viterbi_decode(logits, trans_params, sentence_lengths)
                    tag_preds = test_data.logits_indices_to_tags_seq(viterbi_sequences, sentence_lengths)
                else:
                    tag_preds = test_data.logits_indices_to_tags_seq(logits_indices, sentence_lengths)
                tag_corrects = test_data.logits_indices_to_tags_seq(output_data_indices, sentence_lengths)
                test_prec, test_rec, test_f1 = ChunkEval.compute_f1(tag_preds, tag_corrects)
                print('test precision, recall, f1(chunk): ', test_prec, test_rec, test_f1)

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
    config = Config(args, is_train=True, use_crf=True)
    train(config)
