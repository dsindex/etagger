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

def np_concat(sum_var, var):
    if sum_var is not None: sum_var = np.concatenate((sum_var, var), axis=0)
    else: sum_var = var
    return sum_var

def train_step(sess, model, config, data, summary_op, summary_writer):
    start_time = time.time()
    runopts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    idx = 0
    nbatches = (len(data.sentence_tags) + config.batch_size - 1) // config.batch_size
    prog = Progbar(target=nbatches)
    for ptr in range(0, len(data.sentence_tags), config.batch_size):
        config.is_training = True
        feed_dict={model.input_data_pos_ids: data.sentence_pos_ids[ptr:ptr + config.batch_size],
                   model.input_data_chk_ids: data.sentence_chk_ids[ptr:ptr + config.batch_size],
                   model.output_data: data.sentence_tags[ptr:ptr + config.batch_size],
                   model.is_train: config.is_training,
                   model.sentence_length: data.max_sentence_length}
        feed_dict[model.input_data_word_ids] = data.sentence_word_ids[ptr:ptr + config.batch_size]
        feed_dict[model.input_data_wordchr_ids] = data.sentence_wordchr_ids[ptr:ptr + config.batch_size]
        if config.emb_class == 'elmo':
            feed_dict[model.elmo_input_data_wordchr_ids] = data.sentence_elmo_wordchr_ids[ptr:ptr + config.batch_size]
        if config.emb_class == 'bert':
            feed_dict[model.bert_input_data_token_ids] = data.sentence_bert_token_ids[ptr:ptr + config.batch_size]
            feed_dict[model.bert_input_data_token_masks] = data.sentence_bert_token_masks[ptr:ptr + config.batch_size]
            feed_dict[model.bert_input_data_segment_ids] = data.sentence_bert_segment_ids[ptr:ptr + config.batch_size]
        if config.emb_class == 'bert':
            step, summaries, _, loss, accuracy, learning_rate, bert_embeddings = \
                   sess.run([model.global_step, summary_op, model.train_op, \
                             model.loss, model.accuracy, model.learning_rate, model.bert_embeddings], feed_dict=feed_dict, options=runopts)
            if idx == 0:
                tf.logging.debug('# bert_token_ids')
                t = data.sentence_bert_token_ids[:3]
                tf.logging.debug(' '.join([str(x) for x in np.shape(t)]))
                tf.logging.debug(' '.join([str(x) for x in t]))
                tf.logging.debug('# bert_token_masks')
                t = data.sentence_bert_token_masks[:3]
                tf.logging.debug(' '.join([str(x) for x in np.shape(t)]))
                tf.logging.debug(' '.join([str(x) for x in t]))
                tf.logging.debug('# bert_embedding')
                t = bert_embeddings[:3]
                tf.logging.debug(' '.join([str(x) for x in np.shape(t)]))
                tf.logging.debug(' '.join([str(x) for x in t]))
        else:
            step, summaries, _, loss, accuracy, learning_rate = \
                   sess.run([model.global_step, summary_op, model.train_op, \
                             model.loss, model.accuracy, model.learning_rate], feed_dict=feed_dict, options=runopts)

        summary_writer.add_summary(summaries, step)
        prog.update(idx + 1,
                    [('step', step),
                     ('train loss', loss),
                     ('train accuracy', accuracy),
                     ('lr(invalid if use_bert_optimization)', learning_rate)])
        idx += 1
    duration_time = time.time() - start_time
    out = 'duration_time : ' + str(duration_time) + ' sec for this epoch'
    tf.logging.debug(out)

def dev_step(sess, model, config, data, summary_writer, epoch):
    idx = 0
    nbatches = (len(data.sentence_tags) + config.dev_batch_size - 1) // config.dev_batch_size
    prog = Progbar(target=nbatches)
    sum_loss = 0.0
    sum_accuracy = 0.0
    sum_logits_indices = None
    sum_sentence_lengths = None
    trans_params = None
    global_step = 0
    # evaluate on dev data sliced by dev_batch_size to prevent OOM
    for ptr in range(0, len(data.sentence_tags), config.dev_batch_size):
        config.is_training = False
        feed_dict={model.input_data_pos_ids: data.sentence_pos_ids[ptr:ptr + config.dev_batch_size],
                   model.input_data_chk_ids: data.sentence_chk_ids[ptr:ptr + config.dev_batch_size],
                   model.output_data: data.sentence_tags[ptr:ptr + config.dev_batch_size],
                   model.is_train: config.is_training,
                   model.sentence_length: data.max_sentence_length}
        feed_dict[model.input_data_word_ids] = data.sentence_word_ids[ptr:ptr + config.dev_batch_size]
        feed_dict[model.input_data_wordchr_ids] = data.sentence_wordchr_ids[ptr:ptr + config.dev_batch_size]
        if config.emb_class == 'elmo':
            feed_dict[model.elmo_input_data_wordchr_ids] = data.sentence_elmo_wordchr_ids[ptr:ptr + config.dev_batch_size]
        if config.emb_class == 'bert':
            feed_dict[model.bert_input_data_token_ids] = data.sentence_bert_token_ids[ptr:ptr + config.batch_size]
            feed_dict[model.bert_input_data_token_masks] = data.sentence_bert_token_masks[ptr:ptr + config.batch_size]
            feed_dict[model.bert_input_data_segment_ids] = data.sentence_bert_segment_ids[ptr:ptr + config.batch_size]
        global_step, logits_indices, sentence_lengths, loss, accuracy = \
                 sess.run([model.global_step, model.logits_indices, model.sentence_lengths, \
                           model.loss, model.accuracy], feed_dict=feed_dict)
        prog.update(idx + 1,
                    [('dev loss', loss),
                     ('dev accuracy', accuracy)])
        sum_loss += loss
        sum_accuracy += accuracy
        sum_logits_indices = np_concat(sum_logits_indices, logits_indices)
        sum_sentence_lengths = np_concat(sum_sentence_lengths, sentence_lengths)
        idx += 1
    sum_loss = sum_loss / nbatches
    sum_accuracy = sum_accuracy / nbatches
    sum_output_data_indices = np.argmax(data.sentence_tags, 2)
    tag_preds = data.logits_indices_to_tags_seq(sum_logits_indices, sum_sentence_lengths)
    tag_corrects = data.logits_indices_to_tags_seq(sum_output_data_indices, sum_sentence_lengths)
    tf.logging.debug('[epoch %s/%s] dev precision, recall, f1(token): ' % (epoch, config.epoch))
    token_f1, l_token_prec, l_token_rec, l_token_f1  = TokenEval.compute_f1(config.class_size, sum_logits_indices, sum_output_data_indices, sum_sentence_lengths)
    tf.logging.debug('[' + ' '.join([str(x) for x in l_token_prec]) + ']')
    tf.logging.debug('[' + ' '.join([str(x) for x in l_token_rec]) + ']')
    tf.logging.debug('[' + ' '.join([str(x) for x in l_token_f1]) + ']')
    prec, rec, f1 = ChunkEval.compute_f1(tag_preds, tag_corrects)
    tf.logging.debug('dev precision, recall, f1(chunk): %s, %s, %s' % (prec, rec, f1) + '(invalid for bert due to X tag)')
    chunk_f1 = f1

    # create summaries manually
    summary_value = [tf.Summary.Value(tag='loss', simple_value=sum_loss),
                     tf.Summary.Value(tag='accuracy', simple_value=sum_accuracy),
                     tf.Summary.Value(tag='token_f1', simple_value=token_f1),
                     tf.Summary.Value(tag='chunk_f1', simple_value=chunk_f1)]
    summaries = tf.Summary(value=summary_value)
    summary_writer.add_summary(summaries, global_step)
    
    return token_f1, chunk_f1

def do_train(model, config, train_data, dev_data):
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    feed_dict = {model.wrd_embeddings_init: config.embvec.wrd_embeddings}
    sess.run(tf.global_variables_initializer(), feed_dict=feed_dict) # feed large embedding data
    saver = tf.train.Saver()
    if config.restore is not None:
        saver.restore(sess, config.restore)
        tf.logging.debug('model restored')

    # summary setting
    loss_summary = tf.summary.scalar('loss', model.loss)
    acc_summary = tf.summary.scalar('accuracy', model.accuracy)
    lr_summary = tf.summary.scalar('learning_rate', model.learning_rate)
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, lr_summary])
    train_summary_dir = os.path.join(config.summary_dir, 'summaries', 'train')
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    dev_summary_dir = os.path.join(config.summary_dir, 'summaries', 'dev')
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    early_stopping = EarlyStopping(patience=10, measure='f1', verbose=1)
    max_token_f1 = 0
    max_chunk_f1 = 0
    for e in range(config.epoch):
        train_step(sess, model, config, train_data, train_summary_op, train_summary_writer)
        token_f1, chunk_f1  = dev_step(sess, model, config, dev_data, dev_summary_writer, e)
        # early stopping
        if early_stopping.validate(token_f1, measure='f1'): break
        if token_f1 > max_token_f1 or (max_token_f1 - token_f1 < 0.0005 and chunk_f1 > max_chunk_f1):
            tf.logging.debug('new best f1 score! : %s' % token_f1)
            max_token_f1 = token_f1
            max_chunk_f1 = chunk_f1
            # save best model
            save_path = saver.save(sess, config.checkpoint_dir + '/' + 'ner_model')
            tf.logging.debug('max model saved in file: %s' % save_path)
            tf.train.write_graph(sess.graph, '.', config.checkpoint_dir + '/' + 'graph.pb', as_text=False)
            tf.train.write_graph(sess.graph, '.', config.checkpoint_dir + '/' + 'graph.pb_txt', as_text=True)
            early_stopping.reset(max_token_f1)
        early_stopping.status()
    sess.close()

def train(config):
    # build input data
    train_file = 'data/train.txt'
    dev_file = 'data/dev.txt'
    '''for KOR
    train_file = 'data/kor.train.txt'
    dev_file = 'data/kor.dev.txt'
    '''
    '''for CRZ
    train_file = 'data/cruise.train.txt.in'
    dev_file = 'data/cruise.dev.txt.in'
    '''
    train_data = Input(train_file, config, build_output=True)
    dev_data = Input(dev_file, config, build_output=True)
    tf.logging.debug('loading input data ... done')

    # set for bert optimization
    if config.emb_class == 'bert' and config.use_bert_optimization:
        config.num_train_steps = int((len(train_data.sentence_tags) / config.batch_size) * config.epoch)
        config.num_warmup_steps = int(config.num_train_steps * config.warmup_proportion)

    # create model
    model = Model(config)

    # training
    do_train(model, config, train_data, dev_data)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str, help='path to word embedding vector + vocab(.pkl)', required=True)
    parser.add_argument('--wrd_dim', type=int, help='dimension of word embedding vector', required=True)
    parser.add_argument('--word_length', type=int, default=15, help='max word length')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='dir path to save model(ex, ./checkpoint)')
    parser.add_argument('--restore', type=str, default=None, help='path to saved model(ex, ./checkpoint/ner_model)')
    parser.add_argument('--summary_dir', type=str, default='./runs', help='path to save summary(ex, ./runs)')

    args = parser.parse_args()
    tf.logging.set_verbosity(tf.logging.DEBUG)

    config = Config(args, is_training=True, emb_class='glove', use_crf=True)
    train(config)
