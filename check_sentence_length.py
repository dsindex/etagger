from __future__ import print_function
import sys
import argparse
from input import *

def check(args):
    max_sentence_length = Input.find_max_length('data/train.txt')
    print('train, max_sentence_length = %d' % max_sentence_length)
    max_sentence_length = Input.find_max_length('data/dev.txt')
    print('dev, max_sentence_length = %d' % max_sentence_length)
    max_sentence_length = Input.find_max_length('data/test.txt')
    print('test, max_sentence_length = %d' % max_sentence_length)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    check(parser.parse_args())
