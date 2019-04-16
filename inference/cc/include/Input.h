#ifndef INPUT_H
#define INPUT_H

#include <tensorflow/core/public/session.h>
#include "Config.h"
#include "Vocab.h"

class Input {
  public:
    Input(Config* config, Vocab* vocab, vector<string>& bucket);
    int GetMaxSentenceLength() { return max_sentence_length; }
    tensorflow::Tensor* GetSentenceWordIds() { return sentence_word_ids; }
    tensorflow::Tensor* GetSentenceWordChrIds() { return sentence_wordchr_ids; }
    tensorflow::Tensor* GetSentencePosIds() { return sentence_pos_ids; }
    tensorflow::Tensor* GetSentenceChkIds() { return sentence_chk_ids; }
    tensorflow::Tensor* GetSentenceLength() { return sentence_length; }
    tensorflow::Tensor* GetIsTrain() { return is_train; }
    ~Input();
  
  private:
    // same as input.py
    int max_sentence_length;
    tensorflow::Tensor* sentence_word_ids;    // (1, max_sentence_length)
    tensorflow::Tensor* sentence_wordchr_ids; // (1, max_sentence_length, word_length)
    tensorflow::Tensor* sentence_pos_ids;     // (1, max_sentence_length)
    tensorflow::Tensor* sentence_chk_ids;     // (1, max_sentence_length)
    tensorflow::Tensor* sentence_length;      // scalar tensor
    tensorflow::Tensor* is_train;             // scalar tensor

};

#endif
