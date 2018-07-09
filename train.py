from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle as pkl
from model import Model
from input import *

import sys
import argparse

def train(args):
    '''
    Train model
    '''

    # Build input data
    embvec = pkl.load(open(args.emb_path, 'rb'))
    train_data = Input('data/train.txt', embvec, args.emb_dim, args.class_size, args.sentence_length)
    train_inp = train_data.sentence
    train_out = train_data.sentence_tag
    print('max_sentence_length = %d' % train_data.max_sentence_length)
    dev_data = Input('data/dev.txt', embvec, args.emb_dim, args.class_size, args.sentence_length)
    dev_inp = dev_data.sentence
    dev_out = dev_data.sentence_tag
    print('max_sentence_length = %d' % dev_data.max_sentence_length)
    test_data = Input('data/test.txt', embvec, args.emb_dim, args.class_size, args.sentence_length)
    test_inp = test_data.sentence
    test_out = test_data.sentence_tag
    print('max_sentence_length = %d' % test_data.max_sentence_length)
    print('loading input data ... done')

    # Create model
    args.word_dim = train_data.word_dim
    model = Model(args)

    maximum = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if args.restore is not None:
            saver.restore(sess, args.restore)
            print("model restored")
        for e in range(args.epoch):
            idx = 0
            for ptr in range(0, len(train_inp), args.batch_size):
                print('%s-th batch in %s(size of train_inp)' % (idx, len(train_inp)))
                _, train_loss = sess.run([model.train_op, model.loss], {model.input_data: train_inp[ptr:ptr + args.batch_size],
                                          model.output_data: train_out[ptr:ptr + args.batch_size]})
                print('train loss: %s' % (train_loss))
                idx += 1
            if e % 10 == 0:
                save_path = saver.save(sess, args.checkpoint_dir + '/' + 'model.ckpt')
                print("model saved in file: %s" % save_path)
            pred, length, dev_loss = sess.run([model.prediction, model.length, model.loss], {model.input_data: dev_inp,
                                                                       model.output_data: dev_out})
            print("epoch: %d, dev loss: %s" % (e, dev_loss))
            print('dev score:')
            m = Model.compute_f1(args, pred, dev_out, length)
            if m > maximum:
                maximum = m
                save_path = saver.save(sess, args.checkpoint_dir + '/' + 'model_max.ckpt')
                print("max model saved in file: %s" % save_path)
                pred, length, test_loss = sess.run([model.prediction, model.length, model.loss], {model.input_data: test_inp,
                                                                           model.output_data: test_out})
                print("test score:")
                Model.compute_f1(args, pred, test_out, length)


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
