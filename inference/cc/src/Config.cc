#include "Config.h"

/*
 *  public methods
 */

Config::Config()
{
}

Config::Config(int word_length, bool use_crf)
{
  this->word_length = word_length;
  this->use_crf = use_crf;
}

Config::~Config()
{
}
