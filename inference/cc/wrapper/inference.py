from __future__ import print_function
import sys
import os
import time
import argparse

# etagger
import Etagger

###############################################################################
# nlp : spacy
import spacy
nlp = spacy.load('en')

def get_entity(doc, begin, end):
    for ent in doc.ents:
        # check included
        if ent.start_char <= begin and end <= ent.end_char:
            if ent.start_char == begin: return 'B-' + ent.label_
            else: return 'I-' + ent.label_
    return 'O'
 
def build_bucket(nlp, line):
    bucket = []
    doc = nlp(line)
    for token in doc:
        begin = token.idx
        end   = begin + len(token.text) - 1
        temp = []
        '''
        print(token.i, token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop, begin, end)
        '''
        temp.append(token.text)
        temp.append(token.tag_)
        temp.append('O')     # no chunking info
        entity = get_entity(doc, begin, end)
        temp.append(entity)  # entity by spacy
        temp = ' '.join(temp)
        bucket.append(temp)
    return bucket
###############################################################################

def inference(so_path, frozen_graph_fn, vocab_fn, word_length, lowercase=True, is_memmapped=False):

    etagger = Etagger.initialize(so_path,
                                 frozen_graph_fn,
                                 vocab_fn,
                                 word_length=word_length,
                                 lowercase=lowercase,
                                 is_memmapped=is_memmapped,
                                 num_threads=0)

    num_buckets = 0
    total_duration_time = 0.0
    while 1:
        try: line = sys.stdin.readline()
        except KeyboardInterrupt: break
        if not line: break
        line = line.strip()
        bucket = build_bucket(nlp, line)
        start_time = time.time()
        out = Etagger.analyze(etagger, bucket)
        if not out: continue
        for o in out:
            print(' '.join(o))
        print('')
        duration_time = time.time() - start_time
        out = 'duration_time : ' + str(duration_time) + ' sec'
        sys.stderr.write(out + '\n')
        num_buckets += 1
        total_duration_time += duration_time

    out = 'total_duration_time : ' + str(total_duration_time) + ' sec' + '\n'
    out += 'average processing time / bucket : ' + str(total_duration_time / num_buckets) + ' sec'
    sys.stderr.write(out + '\n')

    Etagger.finalize(etagger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen_graph_fn', type=str, help='path to frozen model(ex, ./exported/ner_frozen.pb)', required=True)
    parser.add_argument('--vocab_fn', type=str, help='path to vocab(ex, vocab.txt)', required=True)
    parser.add_argument('--word_length', type=int, default=15, help='max word length')
    parser.add_argument('--is_memmapped', type=str, default='False', help='is memory mapped graph, True | False')

    args = parser.parse_args()
    is_memmapped = False
    if args.is_memmapped == 'True': is_memmapped = True

    # etagger library path
    so_path = os.path.dirname(os.path.abspath(__file__)) + '/../build' + '/' + 'libetagger.so'

    inference(so_path, args.frozen_graph_fn, args.vocab_fn, args.word_length, lowercase=True, is_memmapped=is_memmapped)
