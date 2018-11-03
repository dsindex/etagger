#include <iostream>
#include <string>
#include "Config.h"

/*
 *  public methods
 */

Config::Config()
{
}

Config::Config(int wrd_dim, int word_length, bool use_crf)
{
  this->wrd_dim = wrd_dim;
  this->word_length = word_length;
  this->use_crf = use_crf;
}

Config::~Config()
{
}
