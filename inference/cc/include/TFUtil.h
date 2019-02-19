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
    tensorflow::Session* CreateSession(int num_threads);
    tensorflow::MemmappedEnv* CreateMemmappedEnv(string graph_fn);
    tensorflow::Session* CreateMemmappedEnvSession(tensorflow::MemmappedEnv* memmapped_env, int num_threads);
    void DestroySession(tensorflow::Session* sess);
    tensorflow::Status LoadFrozenModel(tensorflow::Session *sess, string graph_fn);
    tensorflow::Status LoadFrozenMemmappedModel(tensorflow::MemmappedEnv* memmapped_env, tensorflow::Session *sess, string graph_fn);
    tensorflow::Status LoadModel(tensorflow::Session *sess, string graph_fn, string checkpoint_fn);
    void ViterbiDecode(tensorflow::TTypes<float>::Flat logits_flat,
                       tensorflow::TTypes<float>::Flat trans_params_flat,
                       int max_sentence_length,
                       int class_size,
                       vector<int>& viterbi);
    ~TFUtil();
  
  private:
    void load_lstm_lib();
};

#endif
