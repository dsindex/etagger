#ifndef VOCAB_H
#define VOCAB_H

#include <map>
#include <vector>
#include "Config.h"

class Vocab {

  public:
    Vocab(Config& config);
    bool LoadVocab(std::string vocab_fn);
    ~Vocab();
  
  private:
    Config config;
    std::map<std::string, int> wrd_vocab;
    std::map<std::string, int> chr_vocab;
    std::map<std::string, int> pos_vocab;
    std::map<std::string, int> tag_vocab;
    std::map<int, std::string> itag_vocab;
};

#endif
