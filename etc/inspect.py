from __future__ import print_function
import sys
import argparse

class Inspect:
    def __init__(self):
        self.task = 'inspect'

    def inspect_bucket(self, bucket):
        for line in bucket:
            line = line.replace('\t', ' ')
            tokens = line.split()
            size = len(tokens)
            morph = tokens[0]
            mtag = tokens[1]
            etype = tokens[2]
            tag = tokens[3]
            pred = tokens[4]
            comment = 'SUCC'
            if tag != pred: comment = 'FAIL'
            l = [morph, mtag, etype, tag, pred, comment]
            out = '\t'.join(l)
            print(out)
        print('')

    def inspect(self):
        bucket = []
        while 1:
            try: line = sys.stdin.readline()
            except KeyboardInterrupt: break
            if not line: break
            line = line.strip()
            if not line and len(bucket) >= 1:
                self.inspect_bucket(bucket)
                bucket = []
            if line : bucket.append(line)
        if len(bucket) != 0:
            self.inspect_bucket(bucket)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    i = Inspect()
    i.inspect()
