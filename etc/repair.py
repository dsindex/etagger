from __future__ import print_function
import sys
import argparse

class Repair:
    def __init__(self):
        self.task = 'repair'

    def repair_bucket(self, bucket):
        length = len(bucket)
        for i in range(length):
            line = bucket[i]
            line = line.replace('\t', ' ')
            tokens = line.split()
            word = tokens[0]
            pos = tokens[1]
            chunk = tokens[2]
            tag = tokens[3]
            pred = tokens[4]
            # 'X' -> 'O'
            if pred == 'X': pred = 'O'
            # begining 'I-' -> 'O'
            if pred[:2] == 'I-':
                if i == 0:
                    pred = 'O'
                else:
                    p_line = bucket[i-1]
                    p_line = p_line.replace('\t', ' ')
                    p_tokens = p_line.split()
                    p_pred = p_tokens[4]
                    if p_pred == 'O':
                        pred = 'O'
            l = [word, pos, chunk, tag, pred]
            out = ' '.join(l)
            print(out)
        print('')

    def repair(self):
        bucket = []
        while 1:
            try: line = sys.stdin.readline()
            except KeyboardInterrupt: break
            if not line: break
            line = line.strip()
            if not line and len(bucket) >= 1:
                self.repair_bucket(bucket)
                bucket = []
            if line : bucket.append(line)
        if len(bucket) != 0:
            self.repair_bucket(bucket)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    r = Repair()
    r.repair()
