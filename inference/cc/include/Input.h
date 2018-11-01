#ifndef INPUT_H
#define INPUT_H

#include <tensorflow/core/public/session.h>
#include "Config.h"
#include "Vocab.h"

typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;

class Input {
  public:
    Input(Config& config, Vocab& vocab);
    ~Input();
  
  private:
};

#endif
