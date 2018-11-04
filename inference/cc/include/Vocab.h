#ifndef VOCAB_H
#define VOCAB_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <map>
#include <vector>

using namespace std;

class Vocab {

  public:
    Vocab(string vocab_fn, bool lowercase);
    int GetTagVocabSize() { return tag_vocab.size(); }
    void Split(string s, vector<string>& tokens);
    int GetWid(string word);
    ~Vocab();
  
  private:
    bool LoadVocab(string vocab_fn);
    bool lowercase;
    int pad_wid = 0; // same as config.py
    int unk_wid = 1; 
    int pad_cid = 0;
    int unk_cid = 1;
    int pad_pid = 0;
    int unk_pid = 1;
    int oot_tid = 0;
    string oot_tag = "O"; 
    map<string, int> wrd_vocab;
    map<string, int> chr_vocab;
    map<string, int> pos_vocab;
    map<string, int> tag_vocab;
    map<int, string> itag_vocab;
};

#endif
