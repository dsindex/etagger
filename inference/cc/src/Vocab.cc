#include "Vocab.h"

/*
 *  public methods
 */

Vocab::Vocab(string vocab_fn, bool lowercase=true)
{
  bool loaded = load_vocab(vocab_fn);
  if( !loaded ) {
    throw runtime_error("load_vocab() failed!");
  }
  this->lowercase = lowercase;
}

void Vocab::Split(string s, vector<string>& tokens)
{
  istringstream iss(s);
  for( string ts; iss >> ts; )
    tokens.push_back(ts);
}

int Vocab::GetWid(string word) 
{
  if( this->lowercase ) {
    transform(word.begin(), word.end(), word.begin(),::tolower);
  }
  if( this->wrd_vocab.find(word) != this->wrd_vocab.end() ) {
    return this->wrd_vocab[word];
  }  
  return this->unk_wid;
}

int Vocab::GetCid(string ch)
{
  if( this->chr_vocab.find(ch) != this->chr_vocab.end() ) {
    return this->chr_vocab[ch];
  }
  return this->unk_cid;
}

int Vocab::GetPid(string pos)
{
  if( this->pos_vocab.find(pos) != this->pos_vocab.end() ) {
    return this->pos_vocab[pos];
  }
  return this->unk_pid;
}

int Vocab::GetKid(string chk)
{
  if( this->chk_vocab.find(chk) != this->chk_vocab.end() ) {
    return this->chk_vocab[chk];
  }
  return this->unk_kid;
}

string Vocab::GetTag(int tid)
{
  if( this->itag_vocab.find(tid) != this->itag_vocab.end() ) {
    return this->itag_vocab[tid];
  }
  return this->oot_tag;
}

Vocab::~Vocab()
{
}

/* 
 *  private methods
 */

bool Vocab::load_vocab(string vocab_fn)
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
    if( line.find("# chk_vocab") != string::npos ) mode = 4; // chk_vocab
    if( line.find("# tag_vocab") != string::npos ) mode = 5; // tag_vocab
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
      this->chk_vocab.insert(make_pair(key, id));
    }
    if( mode == 5 ) {
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

