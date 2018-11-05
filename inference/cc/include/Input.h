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
    tensorflow::Tensor* GetSentenceWordChrIds() { return sentence_wordchr_ids; }
    tensorflow::Tensor* GetSentencePosIds() { return sentence_pos_ids; }
    tensorflow::Tensor* GetSentenceEtcs() { return sentence_etcs; }
    tensorflow::Tensor* GetSentenceLength() { return sentence_length; }
    tensorflow::Tensor* GetIsTrain() { return is_train; }
    ~Input();
  
  private:
    int max_sentence_length;
    tensorflow::Tensor* sentence_word_ids;    // (1, max_sentence_length), same as input.py
    tensorflow::Tensor* sentence_wordchr_ids; // (1, max_sentence_length, word_length)
    tensorflow::Tensor* sentence_pos_ids;     // (1, max_sentence_length)
    tensorflow::Tensor* sentence_etcs;        // (1, max_sentence_length, etc_dim)
    tensorflow::Tensor* sentence_length;      // scalar tensor
    tensorflow::Tensor* is_train;             // scalar tensor

    bool is_digits(const string& str); 
    bool is_alphas(const string& str);
    bool is_capital(const char ch);
    bool is_symbol(const char ch);
    void set_shape_vec(string word, vector<float>& shape_vec);
    void set_pos_vec(string pos, vector<float>& pos_vec);
};

#endif
