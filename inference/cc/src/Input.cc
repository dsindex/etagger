#include <iostream>
#include <string>
#include "Input.h"

/*
 *  public methods
 */

Input::Input(Config& config, Vocab& vocab, vector<string>& bucket)
{
  this->max_sentence_length = bucket.size();

  for( int i=0; i < max_sentence_length; i++ ) {
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
    cerr << wid << endl;
    
  }
}

Input::~Input()
{
}
