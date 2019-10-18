from __future__ import print_function
import os
import sys
import time
import argparse
import tensorflow as tf
import numpy as np
from embvec import EmbVec
from config import Config
from model import Model
from input import Input
import feed
from token_eval  import TokenEval
from chunk_eval  import ChunkEval
from progbar import Progbar
from early_stopping import EarlyStopping

def train_step(model, data, summary_op, summary_writer):
    """Train one epoch
    """
    start_time = time.time()
    sess = model.sess
    runopts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    prog = Progbar(target=data.num_batches)
    iterator = data.dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess.run(iterator.initializer)
    for idx in range(data.num_batches):
        try:
            dataset = sess.run(next_element)
        except tf.errors.OutOfRangeError:
            break
        feed_dict = feed.build_feed_dict(model, dataset, data.max_sentence_length, True)
        if 'bert' in model.config.emb_class:
            # compute bert embedding at runtime
            bert_embeddings = sess.run([model.bert_embeddings_subgraph], feed_dict=feed_dict, options=runopts)
            if idx == 0:
                tf.logging.debug('# bert_token_ids')
                t = dataset['bert_token_ids'][:1]
                tf.logging.debug(' '.join([str(x) for x in np.shape(t)]))
                tf.logging.debug(' '.join([str(x) for x in t]))
                tf.logging.debug('# bert_token_masks')
                t = dataset['bert_token_masks'][:1]
                tf.logging.debug(' '.join([str(x) for x in np.shape(t)]))
                tf.logging.debug(' '.join([str(x) for x in t]))
                tf.logging.debug('# bert_wordidx2tokenidx')
                t = dataset['bert_wordidx2tokenidx'][:1]
                tf.logging.debug(' '.join([str(x) for x in np.shape(t)]))
                tf.logging.debug(' '.join([str(x) for x in t]))
            # update feed_dict
            feed_dict[model.bert_embeddings] = feed.align_bert_embeddings(config, bert_embeddings, dataset['bert_wordidx2tokenidx'], idx)
            step, summaries, _, loss, accuracy, f1, learning_rate = \
                sess.run([model.global_step, summary_op, model.train_op, \
                          model.loss, model.accuracy, model.f1, \
                          model.learning_rate], feed_dict=feed_dict, options=runopts)
        else:
            step, summaries, _, loss, accuracy, f1, learning_rate = \
                sess.run([model.global_step, summary_op, model.train_op, \
                          model.loss, model.accuracy, model.f1, \
                          model.learning_rate], feed_dict=feed_dict, options=runopts)

        summary_writer.add_summary(summaries, step)
        prog.update(idx + 1,
                    [('step', step),
                     ('train loss', loss),
                     ('train accuracy', accuracy),
                     ('train f1', f1),
                     ('lr(invalid if use_bert_optimization)', learning_rate)])
    duration_time = time.time() - start_time
    out = '\nduration_time : ' + str(duration_time) + ' sec for this epoch'
    tf.logging.debug(out)

def dev_step(model, data, summary_writer, epoch):
    """Evaluate dev data
    """

    def np_concat(sum_var, var):
        if sum_var is not None: sum_var = np.concatenate((sum_var, var), axis=0)
        else: sum_var = var
        return sum_var

    sess = model.sess
    runopts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    sum_loss = 0.0
    sum_accuracy = 0.0
    sum_f1 = 0.0
    sum_output_indices = None
    sum_logits_indices = None
    sum_sentence_lengths = None
    trans_params = None
    global_step = 0
    prog = Progbar(target=data.num_batches)
    iterator = data.dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess.run(iterator.initializer)

    # evaluate on dev data sliced by batch_size to prevent OOM(Out Of Memory).
    for idx in range(data.num_batches):
        try:
            dataset = sess.run(next_element)
        except tf.errors.OutOfRangeError:
            break
        feed_dict = feed.build_feed_dict(model, dataset, data.max_sentence_length, False)
        if 'bert' in model.config.emb_class:
            # compute bert embedding at runtime
            bert_embeddings = sess.run([model.bert_embeddings_subgraph], feed_dict=feed_dict, options=runopts)
            # update feed_dict
            feed_dict[model.bert_embeddings] = feed.align_bert_embeddings(config, bert_embeddings, dataset['bert_wordidx2tokenidx'], idx)
        global_step, logits_indices, sentence_lengths, loss, accuracy, f1 = \
            sess.run([model.global_step, model.logits_indices, model.sentence_lengths, \
                      model.loss, model.accuracy, model.f1], feed_dict=feed_dict)
        prog.update(idx + 1,
                    [('dev loss', loss),
                     ('dev accuracy', accuracy),
                     ('dev f1', f1)])
        sum_loss += loss
        sum_accuracy += accuracy
        sum_f1 += f1
        sum_output_indices = np_concat(sum_output_indices, np.argmax(dataset['tags'], 2))
        sum_logits_indices = np_concat(sum_logits_indices, logits_indices)
        sum_sentence_lengths = np_concat(sum_sentence_lengths, sentence_lengths)
        idx += 1
    avg_loss = sum_loss / data.num_batches
    avg_accuracy = sum_accuracy / data.num_batches
    avg_f1 = sum_f1 / data.num_batches
    tag_preds = model.config.logits_indices_to_tags_seq(sum_logits_indices, sum_sentence_lengths)
    tag_corrects = model.config.logits_indices_to_tags_seq(sum_output_indices, sum_sentence_lengths)
    tf.logging.debug('\n[epoch %s/%s] dev precision, recall, f1(token): ' % (epoch, model.config.epoch))
    token_f1, l_token_prec, l_token_rec, l_token_f1  = TokenEval.compute_f1(model.config.class_size, 
                                                                            sum_logits_indices,
                                                                            sum_output_indices,
                                                                            sum_sentence_lengths)
    tf.logging.debug('[' + ' '.join([str(x) for x in l_token_prec]) + ']')
    tf.logging.debug('[' + ' '.join([str(x) for x in l_token_rec]) + ']')
    tf.logging.debug('[' + ' '.join([str(x) for x in l_token_f1]) + ']')
    chunk_prec, chunk_rec, chunk_f1 = ChunkEval.compute_f1(tag_preds, tag_corrects)
    tf.logging.debug('dev precision(chunk), recall(chunk), f1(chunk): %s, %s, %s' % \
        (chunk_prec, chunk_rec, chunk_f1) + \
        '(invalid for bert due to X tag)')

    # create summaries manually.
    summary_value = [tf.Summary.Value(tag='loss', simple_value=avg_loss),
                     tf.Summary.Value(tag='accuracy', simple_value=avg_accuracy),
                     tf.Summary.Value(tag='f1', simple_value=avg_f1),
                     tf.Summary.Value(tag='token_f1', simple_value=token_f1),
                     tf.Summary.Value(tag='chunk_f1', simple_value=chunk_f1)]
    summaries = tf.Summary(value=summary_value)
    summary_writer.add_summary(summaries, global_step)
    
    return token_f1, chunk_f1, avg_f1

def fit(model, train_data, dev_data):
    """Do actual training. 
    """

    def get_summary_setting(model):
        config = model.config
        sess   = model.sess
        loss_summary = tf.summary.scalar('loss', model.loss)
        acc_summary = tf.summary.scalar('accuracy', model.accuracy)
        f1_summary = tf.summary.scalar('f1', model.f1)
        lr_summary = tf.summary.scalar('learning_rate', model.learning_rate)
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, f1_summary, lr_summary])
        train_summary_dir = os.path.join(config.summary_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        dev_summary_dir = os.path.join(config.summary_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        return train_summary_op, train_summary_writer, dev_summary_writer

    config = model.config
    sess   = model.sess

    # restore previous model if provided
    saver = tf.train.Saver()
    if config.restore is not None:
        saver.restore(sess, config.restore)
        tf.logging.debug('model restored')

    # summary setting
    train_summary_op, train_summary_writer, dev_summary_writer = get_summary_setting(model)
    
    # train and evaluate
    early_stopping = EarlyStopping(patience=10, measure='f1', verbose=1)
    max_token_f1 = 0
    max_chunk_f1 = 0
    max_avg_f1 = 0
    for e in range(config.epoch):
        train_step(model, train_data, train_summary_op, train_summary_writer)
        token_f1, chunk_f1, avg_f1  = dev_step(model, dev_data, dev_summary_writer, e)
        # early stopping
        if early_stopping.validate(token_f1, measure='f1'): break
        if token_f1 > max_token_f1 or (max_token_f1 - token_f1 < 0.0005 and chunk_f1 > max_chunk_f1):
            tf.logging.debug('new best f1 score! : %s' % token_f1)
            max_token_f1 = token_f1
            max_chunk_f1 = chunk_f1
            max_avg_f1 = avg_f1
            # save best model
            save_path = saver.save(sess, config.checkpoint_dir + '/' + 'ner_model')
            tf.logging.debug('max model saved in file: %s' % save_path)
            tf.train.write_graph(sess.graph, '.', config.checkpoint_dir + '/' + 'graph.pb', as_text=False)
            tf.train.write_graph(sess.graph, '.', config.checkpoint_dir + '/' + 'graph.pb_txt', as_text=True)
            early_stopping.reset(max_token_f1)
        early_stopping.status()
        '''for KOR KMOU
        # early stopping
        if early_stopping.validate(chunk_f1, measure='f1'): break
        if chunk_f1 > max_chunk_f1:
            tf.logging.debug('new best f1 score! : %s' % chunk_f1)
            max_token_f1 = token_f1
            max_chunk_f1 = chunk_f1
            max_avg_f1 = avg_f1
            # save best model
            save_path = saver.save(sess, config.checkpoint_dir + '/' + 'ner_model')
            tf.logging.debug('max model saved in file: %s' % save_path)
            tf.train.write_graph(sess.graph, '.', config.checkpoint_dir + '/' + 'graph.pb', as_text=False)
            tf.train.write_graph(sess.graph, '.', config.checkpoint_dir + '/' + 'graph.pb_txt', as_text=True)
            early_stopping.reset(max_chunk_f1)
        early_stopping.status()
        '''
    sess.close()

def train(config):
    """Prepare input data(train, dev), model and fit
    """

    # build input train and dev data
    train_file = 'data/train.txt'
    dev_file = 'data/dev.txt'
    '''for KOR
    train_file = 'data/kor.train.txt'
    dev_file = 'data/kor.dev.txt'
    '''
    '''for KOR nbest
    train_file = 'data/kor.nbest.train.txt'
    dev_file = 'data/kor.nbest.dev.txt'
    '''
    '''for CRZ
    train_file = 'data/cruise.train.txt.in'
    dev_file = 'data/cruise.dev.txt.in'
    '''
    train_data = Input(train_file, config, build_output=True, do_shuffle=True, reuse=False)
    dev_data = Input(dev_file, config, build_output=True, reuse=False)
    tf.logging.debug('loading input data ... done')
    config.update(train_data)
    tf.logging.debug('config.num_train_steps = %s' % config.num_train_steps)
    tf.logging.debug('config.num_warmup_epoch = %s' % config.num_warmup_epoch)
    tf.logging.debug('config.num_warmup_steps = %s' % config.num_warmup_steps)

    # create model and compile
    model = Model(config)
    model.compile()

    # do actual training
    fit(model, train_data, dev_data)
        
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
