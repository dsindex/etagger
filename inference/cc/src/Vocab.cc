#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "Vocab.h"

using namespace std;

/*
 *  public methods
 */

Vocab::Vocab(string vocab_fn)
{
  bool loaded = LoadVocab(vocab_fn);
  if( !loaded ) {
    throw std::runtime_error("LoadVocab() failed!");
  }
}

int Vocab::GetTagVocabSize()
{
  return this->tag_vocab.size();
}

Vocab::~Vocab()
{
}

void Vocab::Split(string s, vector<string>& tokens)
{
  istringstream iss(s);
  for( string ts; iss >> ts; )
    tokens.push_back(ts);
}

/* 
 *  private methods
 */

bool Vocab::LoadVocab(string vocab_fn)
{
  cerr << "Loading Vocab From " << vocab_fn  << endl; 
  fstream fs(vocab_fn, ios_base::in);
  if( !fs.is_open() ) {
    cerr << "Can't find " << vocab_fn << endl;
    return false;
  }
  string line = "";
  int mode = 0;
  string key = "";
  int id = 0;
  while( getline(fs, line) ) {
    if( line.find("# wrd_vocab") != string::npos ) mode = 1; // wrd_vocab
    if( line.find("# chr_vocab") != string::npos ) mode = 2; // chr_vocab
    if( line.find("# pos_vocab") != string::npos ) mode = 3; // pos_vocab
    if( line.find("# tag_vocab") != string::npos ) mode = 4; // tag_vocab
    vector<string> tokens;
    Split(line, tokens);
    if( tokens.size() != 2 ) continue;
    key = tokens[0];
    id  = atoi(tokens[1].c_str());
    if( mode == 1 ) {
      this->wrd_vocab.insert(make_pair(key, id));
    }
    if( mode == 2 ) {
      this->chr_vocab.insert(make_pair(key, id));
    }
    if( mode == 3 ) {
      this->pos_vocab.insert(make_pair(key, id));
    }
    if( mode == 4 ) {
      this->tag_vocab.insert(make_pair(key, id));
      this->itag_vocab.insert(make_pair(id, key));
    }
  }
  fs.close();

#ifdef DEBUG
  for( auto itr = this->wrd_vocab.cbegin(); itr != this->wrd_vocab.cend(); ++itr ) {
    cout << itr->first << " " << itr->second << endl;
  }
  for( auto itr = this->itag_vocab.cbegin(); itr != this->itag_vocab.cend(); ++itr ) {
    cout << itr->first << " " << itr->second << endl;
  }
#endif

  return true;
}

