from __future__ import print_function
import tensorflow as tf
import numpy as np

import sys
import argparse
from model import Model
from input import *

def f1(args, prediction, target, length):
    '''
    Compute F1 measure
    '''
    tp = np.array([0] * (args.class_size + 1))
    fp = np.array([0] * (args.class_size + 1))
    fn = np.array([0] * (args.class_size + 1))
    target = np.argmax(target, 2)
    prediction = np.argmax(prediction, 2)
    for i in range(len(target)):
        for j in range(length[i]):
            if target[i, j] == prediction[i, j]:
                tp[target[i, j]] += 1
            else:
                fp[target[i, j]] += 1
                fn[prediction[i, j]] += 1
    unnamed_entity = args.class_size - 1
    for i in range(args.class_size):
        if i != unnamed_entity:
            tp[args.class_size] += tp[i]
            fp[args.class_size] += fp[i]
            fn[args.class_size] += fn[i]
    precision = []
    recall = []
    fscore = []
    for i in range(args.class_size + 1):
        precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
        recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
        fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
    print(fscore)
    return fscore[args.class_size]


def train(args):
    '''
    Train model
    '''
    train_inp, train_out = get_train_data()
    test_a_inp, test_a_out = get_test_a_data()
    test_b_inp, test_b_out = get_test_b_data()
    model = Model(args)
    checkpoint_dir = './checkpoint'
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
                save_path = saver.save(sess, checkpoint_dir + '/' + 'model.ckpt')
                print("model saved in file: %s" % save_path)
            pred, length, dev_loss = sess.run([model.prediction, model.length, model.loss], {model.input_data: test_a_inp,
                                                                       model.output_data: test_a_out})
            print("epoch: %d, dev loss: %s" % (e, dev_loss))
            print('test_a score:')
            m = f1(args, pred, test_a_out, length)
            if m > maximum:
                maximum = m
                save_path = saver.save(sess, checkpoint_dir + '/' + 'model_max.ckpt')
                print("max model saved in file: %s" % save_path)
                pred, length, test_loss = sess.run([model.prediction, model.length, model.loss], {model.input_data: test_b_inp,
                                                                           model.output_data: test_b_out})
                print("test_b score:")
                f1(args, pred, test_b_out, length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_dim', type=int, help='dimension of word vector', required=True)
    parser.add_argument('--sentence_length', type=int, help='max sentence length', required=True)
    parser.add_argument('--class_size', type=int, help='number of classes', required=True)
    parser.add_argument('--rnn_size', type=int, default=256, help='hidden dimension of rnn')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in rnn')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--restore', type=str, default=None, help="path of saved model")

    train(parser.parse_args())
