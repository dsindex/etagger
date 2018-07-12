from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle as pkl
from model import Model
from eval  import Eval
from input import *
import sys
import argparse

def train(args):
    # Build input data
    train_file = 'data/train.txt'
    dev_file = 'data/dev.txt'
    test_file = 'data/test.txt'
    #train_file = 'data/eng.train_50'
    #dev_file = 'data/eng.test_a_50'
    #test_file = 'data/eng.test_b_50'
    embvec = pkl.load(open(args.emb_path, 'rb'))
    train_data = Input(train_file, embvec, args.emb_dim, args.class_size, args.sentence_length)
    train_inp_word_ids = train_data.sentence_word_ids
    train_inp_etc = train_data.sentence_etc
    train_out = train_data.sentence_tag
    print('max_sentence_length = %d' % train_data.max_sentence_length)
    dev_data = Input(dev_file, embvec, args.emb_dim, args.class_size, args.sentence_length)
    dev_inp_word_ids = dev_data.sentence_word_ids
    dev_inp_etc = dev_data.sentence_etc
    dev_out = dev_data.sentence_tag
    print('max_sentence_length = %d' % dev_data.max_sentence_length)
    test_data = Input(test_file, embvec, args.emb_dim, args.class_size, args.sentence_length)
    test_inp_word_ids = test_data.sentence_word_ids
    test_inp_etc = test_data.sentence_etc
    test_out = test_data.sentence_tag
    print('max_sentence_length = %d' % test_data.max_sentence_length)
    print('loading input data ... done')

    # Create model
    model = Model(embvec, train_data.etc_dim, args)

    maximum = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if args.restore is not None:
            saver.restore(sess, args.restore)
            print("model restored")
        for e in range(args.epoch):
            idx = 0
            for ptr in range(0, len(train_inp_word_ids), args.batch_size):
                print('%s-th batch in %s(size of train_inp)' % (idx, len(train_inp_word_ids)))
                feed_dict={model.input_data_word_ids: train_inp_word_ids[ptr:ptr + args.batch_size],
                           model.input_data_etc: train_inp_etc[ptr:ptr + args.batch_size],
                           model.output_data: train_out[ptr:ptr + args.batch_size]}
                _, train_loss = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
                print('train loss: %s' % (train_loss))
                idx += 1
            if e % 10 == 0:
                save_path = saver.save(sess, args.checkpoint_dir + '/' + 'model.ckpt')
                print("model saved in file: %s" % save_path)
            feed_dict={model.input_data_word_ids: dev_inp_word_ids,
                       model.input_data_etc: dev_inp_etc,
                       model.output_data: dev_out}
            pred, length, dev_loss = sess.run([model.prediction, model.length, model.loss], feed_dict=feed_dict)
            print("epoch: %d, dev loss: %s" % (e, dev_loss))
            print('dev score:')
            m = Eval.compute_f1(args, pred, dev_out, length)
            if m > maximum:
                maximum = m
                save_path = saver.save(sess, args.checkpoint_dir + '/' + 'model_max.ckpt')
                print("max model saved in file: %s" % save_path)
                feed_dict={model.input_data_word_ids: test_inp_word_ids,
                           model.input_data_etc: test_inp_etc,
                           model.output_data: test_out}
                pred, length, test_loss = sess.run([model.prediction, model.length, model.loss], feed_dict=feed_dict)
                print("test score:")
                Eval.compute_f1(args, pred, test_out, length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str, help='path to word embedding vector(.pkl)', required=True)
    parser.add_argument('--emb_dim', type=int, help='dimension of word embedding vector', required=True)
    parser.add_argument('--sentence_length', type=int, help='max sentence length', required=True)
    parser.add_argument('--class_size', type=int, help='number of classes', required=True)
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='path of saved model(ex, ./checkpoint/model.ckpt)')
    parser.add_argument('--restore', type=str, default=None, help='path to saved model(ex, ./checkpoint/model.ckpt)')

    train(parser.parse_args())
