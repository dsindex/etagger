#include <iostream>
#include <string>
#include "Config.h"

Config::Config()
{
}

Config::Config(int wrd_dim, int word_length, bool use_crf)
{
  this->wrd_dim = wrd_dim;
  this->word_length = word_length;
  this->use_crf = use_crf;
}

void Config::SetClassSize(int class_size)
{
  this->class_size = class_size;
}

int Config::GetClassSize()
{
  return this->class_size;
}

Config::~Config()
{
}
