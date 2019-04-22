#include "Input.h"
#include <cstdio>
#include <cstdlib>

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
    unsigned int* coffarr = build_coffarr(word.c_str(), wlen);
    int cpos_prev = -1;
    string ch = string();
    int index = 0;
    for( int bpos = 0; bpos < wlen && index < word_length; bpos++ ) {
      int cpos = coffarr[bpos];
      if( cpos == cpos_prev ) {
        ch = ch + word[bpos];
      } else {
        if( !ch.empty() ) {
          // 1 character, ex) '가', 'a', '1', '!'
          int cid = vocab->GetCid(ch);
          data_wordchr_ids[i*word_length + index] = cid;
          index += 1;
        }
        ch.clear();
        ch = word[bpos];
      }
      cpos_prev = cpos;
    }
    if( !ch.empty() ) {
      int cid = vocab->GetCid(ch);
      data_wordchr_ids[i*word_length + index] = cid;
      index += 1;
    }
    for( int j = 0; j < word_length - index; j++ ) { // padding cid
      int pad_cid = vocab->GetPadCid();
      data_wordchr_ids[i*word_length + index + j] = pad_cid;
    }
    if( coffarr ) free(coffarr);
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

int Input::utf8_len(char chr)
{
  /*
   *  get utf8 character length
   *
   *  Args:
   *    chr: begining byte in utf8 string.
   *
   *  Returns:
   *    character length for the chr, i.e, range.
   */
  if( (chr & 0x80) == 0x00 )
      return 1;
  else if( (chr & 0xE0) == 0xC0 )
      return 2;
  else if( (chr & 0xF0) == 0xE0 )
      return 3;
  else if( (chr & 0xF8) == 0xF0 )
      return 4;
  else if( (chr & 0xFC) == 0xF8 )
      return 5;
  else if( (chr & 0xFE) == 0xFC )
      return 6;
  else if( (chr & 0xFE ) == 0xFE )
      return 1;
  return 0;
}

unsigned int* Input::build_coffarr(const char* in, int in_size)
{
  /*
   *  compute character offset array
   *  returnd pointer must be released
   *  ex) utf-8 string : 가나다라abcd가나'\0'
   *  -----------------------------------------------------------
   *  0 1 2 3 4 5 6 7 8 9 10 11  12 13 14 15 16 17 18 19 20  21 22
   *  0 0 0 1 1 1 2 2 2 3 3  3   4  5  6  7  8  8  8  9  9   9  10
   *  -----------------------------------------------------------
   * 
   *  usage) 
   *    char* in = "가나다라abcd가나';
   *    int in_size = strlen(in);
   *    unsigned int* coffarr = build_coffarr(in, insize);
   *    int character_pos = coffarr[byte_pos];
   *    if( coffarr ) free(coffarr);
   *  
   *  Args:
   *    in: utf8 string.
   *    in_size: size of in(byte length).
   *
   *  Returns:
   *    unsigned int array. this should be freed later.
   */
    int i, j;
    int index;
    int codelen;
    const char *s = in;
    unsigned int *char_offset_array;

    char_offset_array = (unsigned int*)malloc(sizeof(unsigned int) * (in_size+2));
    if( char_offset_array == NULL ) {
        fprintf(stderr, "char_offset_array : malloc fail!");
        return NULL;
    }
    index=0;
    // compute offset for last '\0'
    for( i = 0; i < in_size+1; i = i + codelen ) {
        codelen = this->utf8_len(s[i]);
        if( codelen == 0 ) {
            fprintf(stderr, "%s contains invalid utf8 begin code", in);
            if( char_offset_array != NULL ) {
                free(char_offset_array);
                return NULL;
            }
        }
        for( j = 0; j < codelen; j++ ) {
            if( codelen == 1 )
                char_offset_array[i] = index;
            else {
                if( j == 0 ) {
                    char_offset_array[i] = index;
                } else {
                    if( this->utf8_len(s[i+j]) == 0 ) { // valid inner code
                        char_offset_array[i+j] = index;
                    } else {
                        fprintf(stderr, "%s contains invalid utf8 inner code", in);
                        free(char_offset_array);
                        return NULL;
                    }
                }
            }
        }
        index++;
    }
    return char_offset_array;
}
