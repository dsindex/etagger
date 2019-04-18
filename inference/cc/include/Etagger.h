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
     *  Args:
     *    frozen_graph_fn: path to a file of frozen graph.
     *    vocab_fn: path to a vocab file.
     *    word_length: max character size of word. ex) 15
     *    lowercase: 1 if vocab file was all lowercased, otherwise 0.
     *    is_memmapped: 1 if frozen graph was memmapped, otherwise 0.
     *    num_threads: number of threads for tensorflow. 0 for all cores, n for n cores.
     *
     *  Python: 
     *    import sys
     *    sys.path.append('path-to/lib')
     *    import ctypes as c
     *    libetagger = c.cdll.LoadLibrary('./libetagger.so')
     *
     *    frozen_graph_fn = 'path-to/ner_frozen.pb'
     *    vocab_fn = 'path-to/vocab.txt'
     *    word_length = c.c_int(15)
     *    lowercase = c.c_int(1)
     *    is_memmapped = c.c_int(1)
     *    num_threads = c.c_int(0)
     *    etagger = libetagger.initialize(frozen_graph_fn, vocab_fn, c.byref(word_length), c.byref(lowercase), c.byref(is_memmapped), c.byref(num_threads))
     */ 
    bool b_lowercase = false;
    if( lowercase ) b_lowercase = true;
    bool b_is_memmapped = false;
    if( is_memmapped ) b_is_memmapped = true;
    return new Etagger(frozen_graph_fn, vocab_fn, word_length, b_lowercase, b_is_memmapped, num_threads);
  } 

  static void split(string s, vector<string>& tokens)
  {
    istringstream iss(s);
    for( string ts; iss >> ts; )
      tokens.push_back(ts);
  }

  int analyze(Etagger* etagger, struct result_obj* robj, int max)
  {
    /*
     *  Args:
     *    etagger: an instance of Etagger , i.e, handler.
     *    robj: list of result_obj.
     *    max:  max size of robj.
     *
     *  Python:
     *    class Result( c.Structure ):
     *        _fields_ = [('word', c.c_char * MAX_WORD ),
     *                    ('pos', c.c_char * MAX_POS ),
     *                    ('chk', c.c_char * MAX_CHK ),
     *                    ('tag', c.c_char * MAX_TAG ),
     *                    ('predict', c.c_char * MAX_TAG )]
     *
     *    bucket = build_bucket(nlp, line)
     *    # ex) bucket
     *    #     word    pos chk tag
     *    #     ...
     *    #     jeju    NNP O   B-GPE
     *    #     island  NN  O   O
     *    #     ...
     *    max_sentence_length = len(bucket)
     *    robj = (Result * max_sentence_length)()
     *    # fill robj from bucket.
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

    // build bucket from robj
    for( int i = 0; i < max; i++ ) {
      string s = string(robj[i].word) + " " + 
                 string(robj[i].pos) + " " + 
                 string(robj[i].chk) + " " + 
                 string(robj[i].tag);
      bucket.push_back(s);
    }

    int ret = etagger->Analyze(bucket);
    if( ret < 0 ) return -1;

    // assign predict to robj
    for( int i = 0; i < max; i++ ) {
      vector<string> tokens;
      split(bucket[i], tokens);
      string predict = tokens[4]; // last one
      strncpy(robj[i].predict, predict.c_str(), MAX_TAG);
    }

    return ret;
  }

  void finalize(Etagger* etagger)
  {
    /*
     *  Args:
     *    etagger: an instance of Etagger , handler
     *  Python:
     *    libetagger.finalize(etagger)
     */
    if( etagger ) {
      delete etagger;
    }
  }

}

#endif
