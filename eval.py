from __future__ import print_function
import numpy as np

import sys
import argparse

cls = {}
tp = {}
fp = {}
fn = {}
precision = {}
recall = {}
fscore = {}

def _eval(bucket):
    for line in bucket:
        tokens = line.split()
        size = len(tokens)
        assert(size == 5)
        w = tokens[0]
        pos = tokens[1]
        chunk = tokens[2]
        tag = tokens[3]
        pred = tokens[4]
        if pred not in tp: tp[pred] = 0
        if pred not in fp: fp[pred] = 0
        if tag not in fn: fn[tag] = 0
        if tag == pred:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[tag] += 1
        cls[pred] = None
        cls[tag] = None

def eval(args):
    bucket = []
    while 1:
        try: line = sys.stdin.readline()
        except KeyboardInterrupt: break
        if not line: break
        line = line.strip()
        if not line and len(bucket) >= 1:
            _eval(bucket)
            bucket = []
        if line : bucket.append(line)
    if len(bucket) != 0:
        _eval(bucket)

    print(tp)
    print(fp)
    print(fn)

    for c, _ in cls.iteritems():
        precision[c] = tp[c]*1.0 / (tp[c] + fp[c])
        recall[c] = tp[c]*1.0 / (tp[c] + fn[c])
        fscore[c] = 2.0*precision[c]*recall[c] / (precision[c] + recall[c])

    print('')
    print('precision:')
    for c, _ in precision.iteritems():
        print(c + ',' + str(precision[c]))
    print('')
    print('recall:')
    for c, _ in recall.iteritems():
        print(c + ',' + str(recall[c]))
    print('')
    print('fscore:')
    for c, _ in fscore.iteritems():
        print(c + ',' + str(fscore[c]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    eval(args)
