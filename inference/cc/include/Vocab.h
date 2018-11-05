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
#include <limits>

using namespace std;

class Vocab {

  public:
    Vocab(string vocab_fn, bool lowercase);
    int GetTagVocabSize() { return tag_vocab.size(); }
    void Split(string s, vector<string>& tokens);
    int GetWid(string word);
    int GetCid(string ch);
    int GetPadCid() { return pad_cid; }
    int GetPid(string pos);
    string GetTag(int tid);
    ~Vocab();
  
  private:
    // same as config.py
    bool lowercase;
    int pad_wid = 0;
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

    bool load_vocab(string vocab_fn);
};

#endif
