#ifndef VOCAB_H
#define VOCAB_H

#include "Config.h"

class Vocab {
  public:
    Vocab();
    Vocab(Config& config, std::string vocab_fn);
    ~Vocab();
  
  private:
};

#endif
