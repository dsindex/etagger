from __future__ import print_function
import sys
import os
import Etagger

import time
import argparse

def inference(frozen_graph_fn, vocab_fn, word_length):

    etagger = Etagger.initialize(frozen_graph_fn,
                                 vocab_fn,
                                 word_length=word_length,
                                 lowercase=True,
                                 is_memmapped=False,
                                 num_threads=0)

    num_buckets = 0
    total_duration_time = 0.0
    bucket = []
    while 1:
        try: line = sys.stdin.readline()
        except KeyboardInterrupt: break
        if not line: break
        line = line.strip()
        out = Etagger.analyze(etagger, line)
        if not out: continue
        for o in out:
            print(' '.join(o).decode('utf-8'))
        print('\n')
        duration_time = time.time() - start_time
        out = 'duration_time : ' + str(duration_time) + ' sec'
        sys.stderr.write(out + '\n')
        num_buckets += 1
        total_duration_time += duration_time

    out = 'total_duration_time : ' + str(total_duration_time) + ' sec' + '\n'
    out += 'average processing time / bucket : ' + str(total_duration_time / num_buckets) + ' sec'
    tf.logging.info(out)

    Etagger.finalize(etagger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen_graph_fn', type=str, help='path to frozen model(ex, ./exported/ner_frozen.pb)', required=True)
    parser.add_argument('--vocab_fn', type=str, help='path to vocab(ex, vocab.txt)', required=True)
    parser.add_argument('--word_length', type=int, default=15, help='max word length')

    args = parser.parse_args()
    inference(args.frozen_graph_fn, args.vocab_fn, args.word_length)
