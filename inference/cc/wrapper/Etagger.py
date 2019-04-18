from __future__ import print_function
import sys
import os
import ctypes as c

libetagger = None

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

def initialize(so_path, frozen_graph_fn, vocab_fn, word_length=15, lowercase=True, is_memmapped=False, num_threads=0):
    global libetagger
    if not libetagger:
        libetagger = c.cdll.LoadLibrary(so_path)
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

def analyze(etagger, bucket):
    global libetagger
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
        out.append([r.word.decode('utf-8'),
                    r.pos.decode('utf-8'),
                    r.chk.decode('utf-8'),
                    r.tag.decode('utf-8'),
                    r.predict.decode('utf-8')])
    return out

def finalize(etagger):
    global libetagger
    libetagger.finalize(etagger)
