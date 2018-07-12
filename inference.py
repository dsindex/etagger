from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle as pkl
from model import Model
from eval  import Eval
from input import *
import sys
import argparse

def inference_bulk(args):
    '''
    inference model by test data
    '''

    embvec = pkl.load(open(args.emb_path, 'rb'))

    # Build input data
    test_file = 'data/test.txt'
    test_data = Input(test_file, embvec, args.emb_dim, args.class_size, args.sentence_length)
    print('max_sentence_length = %d' % test_data.max_sentence_length)
    print('loading input data ... done')

    # Create model
    model = Model(embvec, test_data.etc_dim, args)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, args.restore)
        print("model restored")
        feed_dict = {model.input_data_word_ids: test_data.sentence_word_ids,
                     model.input_data_etc: test_data.sentence_etc,
                     model.output_data: test_data.sentence_tag}
        pred, length, test_loss = sess.run([model.prediction, model.length, model.loss], feed_dict=feed_dict)
        print("test score:")
        Eval.compute_f1(args, pred, test_data.sentence_tag, length)

def inference_interactive(args):
    '''
    inference model interactively
    '''

    embvec = pkl.load(open(args.emb_path, 'rb'))

    # Create model
    etc_dim = 5+5+1
    model = Model(embvec, etc_dim, args)

    sess = tf.Session()
    # Restore model
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, args.restore)
    sys.stderr.write('model restored' +'\n')

    bucket = []
    while 1 :
        try : line = sys.stdin.readline()
        except KeyboardInterrupt : break
        if not line : break
        line = line.strip()
        if not line and len(bucket) >= 1:
            # Build input data
            inp = Input(bucket, embvec, args.emb_dim, args.class_size, args.sentence_length)
            feed_dict = {model.input_data_word_ids: inp.sentence_word_ids,
                         model.input_data_etc: inp.sentence_etc,
                         model.output_data: inp.sentence_tag}
            pred, length, loss = sess.run([model.prediction, model.length, model.loss], feed_dict=feed_dict)
            labels = Input.pred_to_label(pred[0], length[0])
            for i in range(len(bucket)):
                out = bucket[i] + ' ' + labels[i]
                sys.stdout.write(out + '\n')
            sys.stdout.write('\n')
            bucket = []
        if line : bucket.append(line)
    if len(bucket) != 0 :
        # Build input data
        inp = Input(bucket, embvec, args.emb_dim, args.class_size, args.sentence_length)
        feed_dict = {model.input_data_word_ids: inp.sentence_word_ids,
                     model.input_data_etc: inp.sentence_etc,
                     model.output_data: inp.sentence_tag}
        pred, length, loss = sess.run([model.prediction, model.length, model.loss], feed_dict=feed_dict)
        labels = Input.pred_to_label(pred[0], length[0])
        for i in range(len(bucket)):
            out = bucket[i] + ' ' + labels[i]
            sys.stdout.write(out + '\n')
        sys.stdout.write('\n')

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
