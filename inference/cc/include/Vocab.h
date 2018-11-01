#ifndef VOCAB_H
#define VOCAB_H

#include <map>
#include <vector>

class Vocab {

  public:
    Vocab(std::string vocab_fn);
    int GetTagVocabSize();
    void Split(std::string s, std::vector<std::string>& tokens);
    ~Vocab();
  
  private:
    bool LoadVocab(std::string vocab_fn);
    std::map<std::string, int> wrd_vocab;
    std::map<std::string, int> chr_vocab;
    std::map<std::string, int> pos_vocab;
    std::map<std::string, int> tag_vocab;
    std::map<int, std::string> itag_vocab;
};

#endif
