from __future__ import print_function
import sys
import os

# etagger
import ctypes as c
path = os.path.dirname(os.path.abspath(__file__)) + '/../build'
libetagger = c.cdll.LoadLibrary(path + '/' + 'libetagger.so')

# nlp : spacy
import spacy
nlp = spacy.load('en')

# Result class interface to 'struct result_obj'.
# this values should be same as those in 'result_obj.h'.
MAX_WORD = 64
MAX_POS  = 64
MAX_CHK  = 64
MAX_TAG  = 64
class Result( c.Structure ):
    _fields_ = [('word', c.c_char * MAX_WORD ),
                ('pos', c.c_char * MAX_POS ),
                ('chk', c.c_char * MAX_CHK ),
                ('tag', c.c_char * MAX_TAG ),
                ('predict', c.c_char * MAX_TAG )]

def initialize(frozen_graph_fn, vocab_fn, word_length=15, lowercase=True, is_memmapped=False, num_threads=0):
    c_frozen_graph_fn = c.c_char_p(frozen_graph_fn.encode('utf-8')) # unicode -> utf-8
    c_vocab_fn = c.c_char_p(vocab_fn.encode('utf-8')) # unicode -> utf-8
    c_word_length = c.c_int(word_length)
    c_lowercase = c.c_int(0)
    if lowercase == True: c_lowercase = c.c_int(1)
    c_is_memmapped = c.c_int(0)
    if is_memmapped == True: c_is_memmapped = c.c_int(1)
    c_num_threads = c.c_int(num_threads)
    etagger = libetagger.initialize(c_frozen_graph_fn,
                                    c_vocab_fn,
                                    c_word_length,
                                    c_lowercase,
                                    c_is_memmapped,
                                    c_num_threads)
    return etagger

def analyze(etagger, line):
    bucket = build_bucket(nlp, line)
    max_sentence_length = len(bucket)
    robj = (Result * max_sentence_length)() 
    for i in range(max_sentence_length):
        tokens = bucket[i].split()
        robj[i].word = tokens[0].encode('utf-8')
        robj[i].pos = tokens[1].encode('utf-8')
        robj[i].chk = tokens[2].encode('utf-8')
        robj[i].tag = tokens[3].encode('utf-8')
        robj[i].predict = b'O' # initial value 'O'(out of tag)
    c_max_sentence_length = c.c_int(max_sentence_length)
    ret = libetagger.analyze(etagger, c.byref(robj), c_max_sentence_length)
    if ret < 0: return None 
    out = []
    for r in robj:
        out.append([r.word, r.pos, r.chk, r.tag, r.predict])
    return out

def finalize(etagger):
    libetagger.finalize(etagger)

###############################################################################
# nlp : spacy
###############################################################################

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
