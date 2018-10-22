from __future__ import print_function
import sys
import argparse

from flair.data import Sentence
from flair.models import SequenceTagger

def spill_bucket(tagger, bucket):
    sentence = []
    for line in bucket:
        tokens = line.split()
        word = tokens[0]
        sentence.append(word)
    # make a sentence
    sentence = Sentence(' '.join(sentence))
    # run NER over sentence
    tagger.predict(sentence)
    print(sentence)
    print('The following NER tags are found:')
    print(sentence.to_tagged_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # load the NER tagger
    tagger = SequenceTagger.load('ner')

    bucket = []
    while 1:
        try: line = sys.stdin.readline()
        except KeyboardInterrupt: break
        if not line: break
        line = line.strip()
        if not line and len(bucket) >= 1:
            spill_bucket(tagger, bucket)
            bucket = []
        if line : bucket.append(line)

    if len(bucket) != 0:
        spill_bucket(tagger, bucket)
