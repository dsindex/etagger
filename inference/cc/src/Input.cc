#include "Input.h"

/*
 *  public methods
 */

Input::Input(Config* config, Vocab* vocab, vector<string>& bucket)
{
  /*
   *  Args:
   *    config: configuration info. class_size, word_length, etc.
   *    vocab: vocab info. word id, pos id, chk id, tag id, etc.
   *    bucket: list of 'word pos chk tag'.
   */
  this->max_sentence_length = bucket.size();

  // create input tensors
  int word_length = config->GetWordLength();
  tensorflow::TensorShape shape1({1, this->max_sentence_length});
  this->sentence_word_ids = new tensorflow::Tensor(tensorflow::DT_INT32, shape1);
  tensorflow::TensorShape shape2({1, this->max_sentence_length, word_length});
  this->sentence_wordchr_ids = new tensorflow::Tensor(tensorflow::DT_INT32, shape2);
  this->sentence_pos_ids = new tensorflow::Tensor(tensorflow::DT_INT32, shape1);
  this->sentence_chk_ids = new tensorflow::Tensor(tensorflow::DT_INT32, shape1);
  // additional scalar tensor for sentence_length, is_train
  this->sentence_length = new tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape());
  this->is_train = new tensorflow::Tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());

  auto data_word_ids = this->sentence_word_ids->flat<int>().data();
  auto data_wordchr_ids = this->sentence_wordchr_ids->flat<int>().data();
  auto data_pos_ids = this->sentence_pos_ids->flat<int>().data();
  auto data_chk_ids = this->sentence_chk_ids->flat<int>().data();
  auto data_sentence_length = this->sentence_length->flat<int>().data();
  auto data_is_train = this->is_train->flat<bool>().data();
  
  for( int i = 0; i < max_sentence_length; i++ ) {
    string line = bucket[i];
    vector<string> tokens;
    vocab->Split(line, tokens);
    if( tokens.size() != 4 ) {
      throw runtime_error("input tokens must be size 4");
    }
    string word  = tokens[0];
    string pos   = tokens[1];
    string chk   = tokens[2];
    string tag   = tokens[3]; // correct tag(answer) or dummy 'O'
    // build sentence_word_ids
    int wid = vocab->GetWid(word);
    data_word_ids[i] = wid;
    // build sentence_wordchr_ids
    int wlen = word.length();
    for( int j = 0; j < wlen && j < word_length; j++ ) {
      string ch = string() + word[j];
      int cid = vocab->GetCid(ch);
      data_wordchr_ids[i*word_length + j] = cid;
    }
    for( int j = 0; j < word_length - wlen; j++ ) { // padding cid
      int pad_cid = vocab->GetPadCid();
      data_wordchr_ids[i*word_length + wlen + j] = pad_cid;
    }
    // build sentence_pos_ids
    int pid = vocab->GetPid(pos);
    data_pos_ids[i] = pid;
    // build sentence_chk_ids
    int kid = vocab->GetKid(chk);
    data_chk_ids[i] = kid;
  }
  *data_sentence_length = this->max_sentence_length;
  *data_is_train = false;
}

Input::~Input()
{
  if( this->sentence_word_ids ) delete this->sentence_word_ids;
  if( this->sentence_wordchr_ids ) delete this->sentence_wordchr_ids;
  if( this->sentence_pos_ids ) delete this->sentence_pos_ids;
  if( this->sentence_chk_ids ) delete this->sentence_chk_ids;
  if( this->sentence_length ) delete this->sentence_length;
  if( this->is_train ) delete this->is_train;
}

/*
 *  private methods
 */


