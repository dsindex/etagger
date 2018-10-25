from __future__ import print_function
import os
import sys
import random
import time
import argparse
import tensorflow as tf
import numpy as np
from embvec import EmbVec
from config import Config
from model import Model
from input import Input
from token_eval  import TokenEval
from chunk_eval  import ChunkEval
from progbar import Progbar
from early_stopping import EarlyStopping
from viterbi import viterbi_decode

def do_train(model, config, train_data, dev_data):
    early_stopping = EarlyStopping(patience=10, measure='f1', verbose=1)
    maximum = 0
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    runopts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        feed_dict = {}
        if not config.use_elmo: feed_dict = {model.wrd_embeddings_init: config.embvec.wrd_embeddings}
        sess.run(tf.global_variables_initializer(), feed_dict=feed_dict) # feed large embedding data
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
            start_time = time.time()
            idx = 0
            nbatches = (len(train_data.sentence_tags) + config.batch_size - 1) // config.batch_size
            train_prog = Progbar(target=nbatches)
            for ptr in range(0, len(train_data.sentence_tags), config.batch_size):
                feed_dict={model.input_data_pos_ids: train_data.sentence_pos_ids[ptr:ptr + config.batch_size],
                           model.input_data_etcs: train_data.sentence_etcs[ptr:ptr + config.batch_size],
                           model.output_data: train_data.sentence_tags[ptr:ptr + config.batch_size],
                           model.is_train: True,
                           model.sentence_length: train_data.max_sentence_length}
                if config.use_elmo:
                    feed_dict[model.elmo_input_data_wordchr_ids] = train_data.sentence_elmo_wordchr_ids[ptr:ptr + config.batch_size]
                else:
                    feed_dict[model.input_data_word_ids] = train_data.sentence_word_ids[ptr:ptr + config.batch_size]
                    feed_dict[model.input_data_wordchr_ids] = train_data.sentence_wordchr_ids[ptr:ptr + config.batch_size]
                step, train_summaries, _, train_loss, train_accuracy, learning_rate = \
                           sess.run([model.global_step, train_summary_op, model.train_op, \
                                     model.loss, model.accuracy, model.learning_rate], feed_dict=feed_dict, options=runopts)
                train_summary_writer.add_summary(train_summaries, step)
                train_prog.update(idx + 1,
                                  [('step', step),
                                   ('train loss', train_loss),
                                   ('train accuracy', train_accuracy),
                                   ('lr', learning_rate)])
                idx += 1
            duration_time = time.time() - start_time
            out = 'duration_time : ' + str(duration_time) + ' sec for the epoch'
            sys.stderr.write(out + '\n')
            # evaluate on dev data sliced by dev_batch_size to prevent OOM
            idx = 0
            nbatches = (len(dev_data.sentence_tags) + config.dev_batch_size - 1) // config.dev_batch_size
            dev_prog = Progbar(target=nbatches)
            dev_loss = 0.0
            dev_accuracy = 0.0
            dev_logits = None
            dev_logits_indices = None
            dev_trans_params = None
            dev_output_data_indices = None
            dev_sentence_lengths = None
            for ptr in range(0, len(dev_data.sentence_tags), config.dev_batch_size):
                feed_dict={model.input_data_pos_ids: dev_data.sentence_pos_ids[ptr:ptr + config.dev_batch_size],
                           model.input_data_etcs: dev_data.sentence_etcs[ptr:ptr + config.dev_batch_size],
                           model.output_data: dev_data.sentence_tags[ptr:ptr + config.dev_batch_size],
                           model.is_train: False,
                           model.sentence_length: dev_data.max_sentence_length}
                if config.use_elmo:
                    feed_dict[model.elmo_input_data_wordchr_ids] = dev_data.sentence_elmo_wordchr_ids[ptr:ptr + config.dev_batch_size]
                else:
                    feed_dict[model.input_data_word_ids] = dev_data.sentence_word_ids[ptr:ptr + config.dev_batch_size]
                    feed_dict[model.input_data_wordchr_ids] = dev_data.sentence_wordchr_ids[ptr:ptr + config.dev_batch_size]
                step, summaries, logits, logits_indices, trans_params, \
                    output_data_indices, sentence_lengths, loss, accuracy = \
                           sess.run([model.global_step, dev_summary_op, model.logits, model.logits_indices, \
                                     model.trans_params, model.output_data_indices, model.sentence_lengths, \
                                     model.loss, model.accuracy], feed_dict=feed_dict)
                # FIXME how to write dev_loss, dev_accuracy to summary?
                if ptr == 0: dev_summary_writer.add_summary(summaries, step)
                dev_prog.update(idx + 1,
                                [('dev loss', loss),
                                 ('dev accuracy', accuracy)])
                dev_loss += loss
                dev_accuracy += accuracy
                if dev_logits is not None: dev_logits = np.concatenate((dev_logits, logits), axis=0)
                else: dev_logits = logits
                if dev_logits_indices is not None: dev_logits_indices = np.concatenate((dev_logits_indices, logits_indices), axis=0)
                else: dev_logits_indices = logits_indices
                if dev_trans_params is None: dev_trans_params = trans_params
                if dev_output_data_indices is not None: dev_output_data_indices = np.concatenate((dev_output_data_indices, output_data_indices), axis=0)
                else: dev_output_data_indices = output_data_indices
                if dev_sentence_lengths is not None: dev_sentence_lengths = np.concatenate((dev_sentence_lengths, sentence_lengths), axis=0)
                else: dev_sentence_lengths = sentence_lengths
                idx += 1
            dev_loss = dev_loss // nbatches
            dev_accuracy = dev_accuracy // nbatches
            print('[epoch %s/%s] dev precision, recall, f1(token): ' % (e, config.epoch))
            token_f1 = TokenEval.compute_f1(config.class_size, dev_logits, dev_data.sentence_tags, dev_sentence_lengths)
            ''' 
            if config.use_crf:
                viterbi_sequences = viterbi_decode(dev_logits, dev_trans_params, dev_sentence_lengths)
                tag_preds = dev_data.logits_indices_to_tags_seq(viterbi_sequences, dev_sentence_lengths)
            else:
                tag_preds = dev_data.logits_indices_to_tags_seq(dev_logits_indices, dev_sentence_lengths)
            tag_corrects = dev_data.logits_indices_to_tags_seq(dev_output_data_indices, dev_sentence_lengths)
            dev_prec, dev_rec, dev_f1 = ChunkEval.compute_f1(tag_preds, tag_corrects)
            print('dev precision, recall, f1(chunk): ', dev_prec, dev_rec, dev_f1)
            chunk_f1 = dev_f1
            m = chunk_f1
            '''
            m = token_f1
            # early stopping
            if early_stopping.validate(m, measure='f1'): break
            if m > maximum:
                print('new best f1 score! : %s' % m)
                maximum = m
                # save best model
                save_path = saver.save(sess, config.checkpoint_dir + '/' + 'ner_model')
                print('max model saved in file: %s' % save_path)
                tf.train.write_graph(sess.graph, '.', config.checkpoint_dir + '/' + 'graph.pb', as_text=False)
                tf.train.write_graph(sess.graph, '.', config.checkpoint_dir + '/' + 'graph.pb_txt', as_text=True)
    sess.close()

def train(config):
    # build input data
    train_file = 'data/train.txt'
    dev_file = 'data/dev.txt'
    train_data = Input(train_file, config, build_output=True)
    dev_data = Input(dev_file, config, build_output=True)
    print('loading input data ... done')

    # create model
    model = Model(config)

    # training
    do_train(model, config, train_data, dev_data)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str, help='path to word embedding vector(.pkl)', required=True)
    parser.add_argument('--wrd_dim', type=int, help='dimension of word embedding vector', required=True)
    parser.add_argument('--word_length', type=int, default=15, help='max word length')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='dir path to save model(ex, ./checkpoint)')
    parser.add_argument('--restore', type=str, default=None, help='path to saved model(ex, ./checkpoint/ner_model)')
    parser.add_argument('--summary_dir', type=str, default='./runs', help='path to save summary(ex, ./runs)')

    args = parser.parse_args()
    config = Config(args, arg_train=True, use_elmo=False, use_crf=True)
    train(config)
