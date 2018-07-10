from __future__ import print_function
import numpy as np

import sys
import argparse


class Eval:
    '''
    Evaluation class, ex) compute precision, recall, fscore
    '''

    def __init__(self):
        self.cls = {}
        self.tp = {}
        self.fp = {}
        self.fn = {}
        self.precision = {}
        self.recall = {}
        self.fscore = {}

    def eval_bucket(self, bucket):
        for line in bucket:
            tokens = line.split()
            size = len(tokens)
            assert(size == 5)
            w = tokens[0]
            pos = tokens[1]
            chunk = tokens[2]
            tag = tokens[3]
            pred = tokens[4]
            if pred not in self.tp: self.tp[pred] = 0
            if pred not in self.fp: self.fp[pred] = 0
            if tag not in self.fn: self.fn[tag] = 0
            if tag == pred:
                self.tp[pred] += 1
            else:
                self.fp[pred] += 1
                self.fn[tag] += 1
            self.cls[pred] = None
            self.cls[tag] = None

    def eval(self):
        bucket = []
        while 1:
            try: line = sys.stdin.readline()
            except KeyboardInterrupt: break
            if not line: break
            line = line.strip()
            if not line and len(bucket) >= 1:
                self.eval_bucket(bucket)
                bucket = []
            if line : bucket.append(line)
        if len(bucket) != 0:
            self.eval_bucket(bucket)

        print(self.tp)
        print(self.fp)
        print(self.fn)

        for c, _ in self.cls.iteritems():
            self.precision[c] = self.tp[c]*1.0 / (self.tp[c] + self.fp[c])
            self.recall[c] = self.tp[c]*1.0 / (self.tp[c] + self.fn[c])
            self.fscore[c] = 2.0*self.precision[c]*self.recall[c] / (self.precision[c] + self.recall[c])

        print('')
        print('precision:')
        for c, _ in self.precision.iteritems():
            print(c + ',' + str(self.precision[c]))
        print('')
        print('recall:')
        for c, _ in self.recall.iteritems():
            print(c + ',' + str(self.recall[c]))
        print('')
        print('fscore:')
        for c, _ in self.fscore.iteritems():
            print(c + ',' + str(self.fscore[c]))

    @staticmethod
    def compute_f1(args, prediction, target, length):
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
                    tp[prediction[i, j]] += 1
                else:
                    fp[prediction[i, j]] += 1
                    fn[target[i, j]] += 1
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
        print('precision, recall, fscore')
        print(precision)
        print(recall)
        print(fscore)
        return fscore[args.class_size]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    ev = Eval()
    ev.eval()
