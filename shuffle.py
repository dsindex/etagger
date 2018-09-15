import sys
import argparse
import random

__all__ = [
    'Shuffle'
]

class Shuffle :
    def __init__(self) :
        self.buckets = []

    def spill_bucket(self, bucket) :
        self.buckets.append(bucket)

    def add(self, path) :
        fd = open(path, 'r')
        bucket = []
        while 1 :
            try : line = fd.readline()
            except KeyboardInterrupt : break
            if not line : break
            line = line.strip()
            if not line and len(bucket) >= 1 :
                self.spill_bucket(bucket)
                bucket = []
            if line : bucket.append(line)
        if len(bucket) != 0 :
            self.spill_bucket(bucket)
        fd.close()

    def shuffle(self, path) :
        fd = open(path, 'w')
        random.shuffle(self.buckets)
        for bucket in self.buckets :
            for line in bucket :
                fd.write(line + '\n')
            fd.write('\n')
        fd.close()

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    path = 'data/train.txt'
    sp = Shuffle()
    sp.add(path)
    sp.shuffle(path + '.shuffle')

