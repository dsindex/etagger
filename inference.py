from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle as pkl
from model import Model
from input import *

import sys
import argparse

def inference(args):
    '''
    inference model
    '''

    # Build input data
    embvec = pkl.load(open(args.emb_path, 'rb'))
    test_data = Input('data/test.txt', embvec, args.emb_dim, args.class_size, args.sentence_length)
    test_inp = test_data.sentence
    test_out = test_data.sentence_tag
    print('max_sentence_length = %d' % test_data.max_sentence_length)
    print('loading input data ... done')

    # Create model
    args.word_dim = test_data.word_dim
    model = Model(args)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, args.restore)
        print("model restored")
        pred, length, test_loss = sess.run([model.prediction, model.length, model.loss], {model.input_data: test_inp, model.output_data: test_out})
        print("test score:")
        Model.f1(args, pred, test_out, length)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str, help='path to word embedding vector(.pkl)', required=True)
    parser.add_argument('--emb_dim', type=int, help='dimension of word embedding vector', required=True)
    parser.add_argument('--sentence_length', type=int, help='max sentence length', required=True)
    parser.add_argument('--class_size', type=int, help='number of classes', required=True)
    parser.add_argument('--restore', type=str, help='path to saved model(ex, ./checkpoint/model_max.ckpt)', required=True)

    inference(parser.parse_args())
