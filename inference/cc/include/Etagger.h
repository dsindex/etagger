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

#endif
