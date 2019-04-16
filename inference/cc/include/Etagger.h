#ifndef ETAGGER_H
#define ETAGGER_H

#include "TFUtil.h"
#include "Input.h"


class Etagger {
  public:
    Etagger(string frozen_graph_fn, string vocab_fn, int word_length, bool lowercase, bool is_memmapped, int num_threads);
    int Analyze(vector<string>& bucket);
    ~Etagger();
  
  private:
    TFUtil* util;
    tensorflow::Session* sess;
    Config* config;
    Vocab* vocab;

};

#include "result_obj.h" // for c, python wrapper

extern "C" {

  Etagger* initialize(const char* frozen_graph_fn,
                      const char* vocab_fn,
                      int word_length,
                      int lowercase,
                      int is_memmapped,
                      int num_threads)
  {
    /*
     *  Python: 
     *    import sys
     *    sys.path.append('path-to/lib')
     *    import ctypes as c
     *    libetagger = c.cdll.LoadLibrary( './libetagger.so' )
     *
     *    frozen_graph_fn = 'path-to/ner_frozen.pb'
     *    vocab_fn = 'path-to/vocab.txt'
     *    word_length = c.c_int( 15 )
     *    lowercase = c.c_int(1)
     *    is_memmapped = c.c_int(0)
     *    num_threads = c.c_int(0)
     *    etagger = libetagger.initialize(frozen_graph_fn, vocab_fn, c.byref(word_length), c.byref(lowercase), c.byref(is_memmapped), c.byref(num_threads))
     */ 
    bool b_lowercase = false;
    if( lowercase ) b_lowercase = true;
    bool b_is_memmapped = false;
    if( is_memmapped ) b_is_memmapped = true;
    return new Etagger(frozen_graph_fn, vocab_fn, word_length, b_lowercase, b_is_memmapped, num_threads);
  } 

  int analyze(Etagger* etagger, struct result_obj* robj)
  {
    /*
     *  Python:
     *    class Result( c.Structure ):
     *        _fields_ = [('word', c.c_char * MAX_WORD ),
     *                    ('pos', c.c_char * MAX_POS ),
     *                    ('chk', c.c_char * MAX_CHK ),
     *                    ('tag', c.c_char * MAX_TAG ),
     *                    ('predict', c.c_char * MAX_TAG )]
     *
     *    robj = (Result * max_sentence_length)()
     *    # fill robj from bucket.
     *    # ex) bucket = build_bucket(spacy_nlp, text)
     *    #     word    pos chk tag
     *    #     ...
     *    #     jeju    NNP O   B-GPE
     *    #     island  NN  O   O
     *    #     ...
     *    for i in range(max_sentence_length):
     *        tokens = bucket[i].split()
     *        robj[i].word = tokens[0]
     *        robj[i].pos = tokens[1]
     *        robj[i].chk = tokens[2]
     *        robj[i].tag = tokens[3]
     *        robj[i].predict = 'O'
     *    ret = libetagger.analyze(etagger, c.byref(robj))
     *    for r in robj:
     *        print(r.word, r.pos, r.chk, r.tag, r.predict)
     *
     *  Returns:
     *    number of tokens.
     *    -1 if failed.   
     *    analyzed results are saved to robj itself.
     */
    vector<string> bucket;
    // TODO : build bucket from robj

    int ret = etagger->Analyze(bucket);
    if( ret < 0 ) return -1;

    // TODO : assign tags to robj

    return ret;
  }

  void finalize(Etagger* etagger)
  {
    /*
     *  python:
     *    libetagger.finalize(etagger)
     */
    if( etagger ) {
      delete etagger;
    }
  }

}

#endif
