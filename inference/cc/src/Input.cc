#include "Input.h"

/*
 *  public methods
 */

Input::Input(Config& config, Vocab& vocab, vector<string>& bucket)
{
  this->max_sentence_length = bucket.size();

  // create input tensors
  int word_length = config.GetWordLength();
  int etc_dim = config.GetEtcDim();
  tensorflow::TensorShape shape1({1, this->max_sentence_length});
  this->sentence_word_ids = new tensorflow::Tensor(tensorflow::DT_INT32, shape1);
  tensorflow::TensorShape shape2({1, this->max_sentence_length, word_length});
  this->sentence_wordchr_ids = new tensorflow::Tensor(tensorflow::DT_INT32, shape2);
  this->sentence_pos_ids = new tensorflow::Tensor(tensorflow::DT_INT32, shape1);
  tensorflow::TensorShape shape3({1, this->max_sentence_length, etc_dim});
  this->sentence_etcs = new tensorflow::Tensor(tensorflow::DT_FLOAT, shape3);
  // additional scalar tensor for sentence_length, is_train
  this->sentence_length = new tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape());
  this->is_train = new tensorflow::Tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());

  auto data_word_ids = this->sentence_word_ids->flat<int>().data();
  auto data_wordchr_ids = this->sentence_wordchr_ids->flat<int>().data();
  auto data_pos_ids = this->sentence_pos_ids->flat<int>().data();
  auto data_etcs = this->sentence_etcs->flat<float>().data();
  auto data_sentence_length = this->sentence_length->flat<int>().data();
  auto data_is_train = this->is_train->flat<bool>().data();
  
  for( int i = 0; i < max_sentence_length; i++ ) {
    string line = bucket[i];
    vector<string> tokens;
    vocab.Split(line, tokens);
    if( tokens.size() != 4 ) {
      throw runtime_error("input tokens must be size 4");
    }
    string word  = tokens[0];
    string pos   = tokens[1];
    string chunk = tokens[2];
    string tag   = tokens[3];
    // build sentence_word_ids
    int wid = vocab.GetWid(word);
    data_word_ids[i] = wid;
    // build sentence_wordchr_ids
    int wlen = word.length();
    for( int j = 0; j < wlen; j++ ) {
      string ch = string() + word[j];
      int cid = vocab.GetCid(ch);
      data_wordchr_ids[i*word_length + j] = cid;
    }
    for( int j = 0; j < word_length - wlen; j++ ) { // padding cid
      int pad_cid = vocab.GetPadCid();
      data_wordchr_ids[i*word_length + wlen + j] = pad_cid;
    }
    // build sentence_pos_ids
    int pid = vocab.GetPid(pos);
    data_pos_ids[i] = pid;
    // build sentence_etcs
    vector<float> shape_vec(9, 0); // same as input.py
    set_shape_vec(word, shape_vec);
    vector<float> pos_vec(5, 0);   // same as input.py
    set_pos_vec(pos, pos_vec);
    for( int j = 0; j < 9; j++ ) {
      data_etcs[i*etc_dim + j] = shape_vec[j];
    } 
    for( int j = 0; j < 5; j++ ) {
      data_etcs[i*etc_dim + 9 + j] = pos_vec[j];
    } 
  }
  *data_sentence_length = this->max_sentence_length;
  *data_is_train = false;
}

Input::~Input()
{
  if( this->sentence_word_ids ) delete this->sentence_word_ids;
  if( this->sentence_wordchr_ids ) delete this->sentence_wordchr_ids;
  if( this->sentence_pos_ids ) delete this->sentence_pos_ids;
  if( this->sentence_etcs ) delete this->sentence_etcs;
  if( this->sentence_length ) delete this->sentence_length;
  if( this->is_train ) delete this->is_train;
}

/*
 *  private methods
 */

bool Input::is_digits(const string& str)
{
    return all_of(str.begin(), str.end(), ::isdigit);
}

bool Input::is_alphas(const string& str)
{
    return all_of(str.begin(), str.end(), ::isalpha);
}

bool Input::is_capital(const char ch)
{
  if( ch >= 'A' && ch <= 'Z' ) return true;
  return false;
}

bool Input::is_symbol(const char ch)
{
  if( ! ::isalpha(ch) && ! ::isdigit(ch) ) return true;
  return false;
}

void Input::set_shape_vec(string word, vector<float>& shape_vec)
{
  // shape_vec : 9, same as input.py
  // build language specific features:
  //   no-info[0], allDigits[1], mixedDigits[2], allSymbols[3],
  //   mixedSymbols[4], upperInitial[5], lowercase[6], allCaps[7], mixedCaps[8]

  int size = word.length();
  if( is_digits(word) ) {
    shape_vec[1] = 1; // allDigits
  } else if( is_alphas(word) ) {
    int n_caps = 0;
    for( int i = 0; i < size; i++ ) {
      if( is_capital(word[i]) ) n_caps += 1;
    }
    if( n_caps == 0 ) {
      shape_vec[6] = 1; // lowercase
    } else {
      if( size == n_caps ) {
        shape_vec[7] = 1; // allCaps
      } else {
        if( is_capital(word[0]) )
          shape_vec[5] = 1; // upperInitial
        else
          shape_vec[8] = 1; // mixedCaps 
      }
    }
  } else {
    int n_digits = 0;
    int n_symbols = 0;
    for( int i = 0; i < size; i++ ) {
      if( ::isdigit(word[i]) ) n_digits += 1;
      if( is_symbol(word[i]) )  n_symbols += 1;
    }
    if( n_digits >= 1 ) shape_vec[2] = 1; // mixedDigits
    if( n_symbols > 0 ) {
      if( size == n_symbols ) shape_vec[3] = 1; // allSymbols
      else shape_vec[4] = 1; // mixedSymbols
    }
    if( n_digits == 0 && n_symbols == 0 )
      shape_vec[0] = 1; // no-info
  }
}

void Input::set_pos_vec(string pos, vector<float>& pos_vec)
{
  // pos_vec : 5, same as input.py
  if( pos == "NN" || pos == "NNS" ) pos_vec[0] = 1;
  else if( pos == "FW" ) pos_vec[1] = 1;
  else if( pos == "NNP" || pos == "NNPS" ) pos_vec[2] = 1;
  else if( pos == "VB" ) pos_vec[3] = 1;
  else pos_vec[4] = 1;
}

