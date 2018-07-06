from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle as pkl
from model import Model
from input import *

import sys
import argparse

def inference_bulk(args):
    '''
    inference model by test data
    '''

    embvec = pkl.load(open(args.emb_path, 'rb'))

    # Build input data
    test_data = Input()
    test_data.create_input_bulk('data/test.txt', embvec, args.emb_dim, args.class_size, args.sentence_length)
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

def inference_interactive(args):
    '''
    inference model interactively
    '''

    embvec = pkl.load(open(args.emb_path, 'rb'))

    # Create model
    args.word_dim = args.emb_dim + 11
    model = Model(args)

    sess = tf.Session()
    # Restore model
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, args.restore)
    print("model restored")

    bucket = []
    while 1 :
        try : line = sys.stdin.readline()
        except KeyboardInterrupt : break
        if not line : break
        line = line.strip()
        if not line and len(bucket) >= 1:
            # Build input data
            inp = Input()
            inp.create_input_interactive(bucket, embvec, args.emb_dim, args.sentence_length)
            pred, length, loss = sess.run([model.prediction, model.length, model.loss], {model.input_data: inp.sentence})
            print(pred)
            bucket = []
        if line : bucket.append(line)
    if len(bucket) != 0 :
        # Build input data
        inp = Input()
        inp.create_input_interactive(bucket, embvec, args.emb_dim, args.sentence_length)
        pred, length, loss = sess.run([model.prediction, model.length, model.loss], {model.input_data: inp.sentence})
        print(pred)
		
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str, help='path to word embedding vector(.pkl)', required=True)
    parser.add_argument('--emb_dim', type=int, help='dimension of word embedding vector', required=True)
    parser.add_argument('--sentence_length', type=int, help='max sentence length', required=True)
    parser.add_argument('--class_size', type=int, help='number of classes', required=True)
    parser.add_argument('--restore', type=str, help='path to saved model(ex, ./checkpoint/model_max.ckpt)', required=True)
    parser.add_argument('--interactive', type=int, default=0, help='interactive mode')

    args = parser.parse_args()
    if args.interactive:
        inference_interactive(args)
    else:
        inference_bulk(args)
