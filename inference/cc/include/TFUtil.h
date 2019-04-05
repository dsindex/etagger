#ifndef TFUTIL_H
#define TFUTIL_H

#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/c/c_api.h>
#include <tensorflow/core/util/memmapped_file_system.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

using namespace std;

typedef vector<pair<string, tensorflow::Tensor>> tensor_dict;

class TFUtil {
  
  public:
    TFUtil();
    tensorflow::MemmappedEnv* CreateMemmappedEnv(string graph_fn);
    tensorflow::Session* CreateSession(tensorflow::MemmappedEnv* memmapped_env, int num_threads);
    void DestroySession(tensorflow::Session* sess);
    tensorflow::Status LoadFrozenModel(tensorflow::Session *sess, string graph_fn);
    tensorflow::Status LoadFrozenMemmappedModel(tensorflow::MemmappedEnv* memmapped_env, tensorflow::Session *sess);
    tensorflow::Status LoadModel(tensorflow::Session *sess, string graph_fn, string checkpoint_fn);
    ~TFUtil();
  
  private:
    void load_lstm_lib();
    void load_qrnn_lib();
};

#endif
