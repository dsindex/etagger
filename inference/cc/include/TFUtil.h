#ifndef TFUTIL_H
#define TFUTIL_H

#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/c/c_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

using namespace std;

typedef vector<pair<string, tensorflow::Tensor>> tensor_dict;

class TFUtil {
  
  public:
    TFUtil();
    tensorflow::Session* CreateSession();
    void DestroySession(tensorflow::Session* sess);
    void LoadLSTMLibrary();
    tensorflow::Status LoadFrozenModel(tensorflow::Session *sess, string graph_fn);
    tensorflow::Status LoadModel(tensorflow::Session *sess, string graph_fn, string checkpoint_fn);
    ~TFUtil();
  
  private:
};

#endif
