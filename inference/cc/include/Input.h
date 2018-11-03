#ifndef INPUT_H
#define INPUT_H

#include "Config.h"
#include "Vocab.h"

class Input {
  public:
    Input(Config& config, Vocab& vocab, vector<string>& bucket);
    int GetMaxSentenceLength() { return max_sentence_length; }
    ~Input();
  
  private:
    int max_sentence_length;
    vector<int> sentence_word_ids;
    vector<vector<int>> sentence_wordchr_ids;
    vector<int> sentence_pos_ids;
    vector<vector<int>> sentence_etcs;
};

#endif
