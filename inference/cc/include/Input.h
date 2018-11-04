#ifndef INPUT_H
#define INPUT_H

#include <tensorflow/core/public/session.h>
#include "Config.h"
#include "Vocab.h"

class Input {
  public:
    Input(Config& config, Vocab& vocab, vector<string>& bucket);
    int GetMaxSentenceLength() { return max_sentence_length; }
    tensorflow::Tensor* GetSentenceWordIds() { return sentence_word_ids; }
    ~Input();
  
  private:
    int max_sentence_length;
    tensorflow::Tensor* sentence_word_ids;    // (1, max_sentence_length), same as input.py
    tensorflow::Tensor* sentence_wordchr_ids; // (1, max_sentence_length, word_length)
    tensorflow::Tensor* sentence_pos_ids;     // (1, max_sentence_length)
    tensorflow::Tensor* sentence_etcs;        // (1, max_sentence_length, etc_dim)
};

#endif
