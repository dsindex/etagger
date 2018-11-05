#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/c/c_api.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include "Input.h"
#include <iostream>
#include <string>

typedef vector<pair<string, tensorflow::Tensor>> tensor_dict;

void LoadLSTMLibrary() {
  // Load _lstm_ops.so library(from LB_LIBRARY_PATH) for LSTMBlockFusedCell()
  TF_Status* status = TF_NewStatus();
  TF_LoadLibrary("_lstm_ops.so", status);
  if( TF_GetCode(status) != TF_OK ) {
    throw runtime_error("fail to load _lstm_ops.so");
  }
  TF_DeleteStatus(status);
}

tensorflow::Status LoadModel(tensorflow::Session *sess, string graph_fn,
                             string checkpoint_fn = "") {
  tensorflow::Status status;

  // Read in the protobuf graph we exported
  tensorflow::MetaGraphDef graph_def;
  status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
  if( status != tensorflow::Status::OK() ) return status;

  // Create the graph in the current session
  status = sess->Create(graph_def.graph_def());
  if( status != tensorflow::Status::OK() ) return status;

  // Restore model from checkpoint, iff checkpoint is given
  if( checkpoint_fn != "" ) {
    const string restore_op_name = graph_def.saver_def().restore_op_name();
    const string filename_tensor_name =
        graph_def.saver_def().filename_tensor_name();

    tensorflow::Tensor filename_tensor(tensorflow::DT_STRING,
                                       tensorflow::TensorShape());
    filename_tensor.scalar<string>()() = checkpoint_fn;

    tensor_dict feed_dict = {{filename_tensor_name, filename_tensor}};
    status = sess->Run(feed_dict, {}, {restore_op_name}, nullptr);
    if( status != tensorflow::Status::OK() ) return status;
  } else {
    // virtual Status Run(const vector<pair<string, Tensor> >& inputs,
    //                  const vector<string>& output_tensor_names,
    //                  const vector<string>& target_node_names,
    //                  vector<Tensor>* outputs) = 0;
    status = sess->Run({}, {}, {"init"}, nullptr);
    if( status != tensorflow::Status::OK() ) return status;
  }

  return tensorflow::Status::OK();
}

tensorflow::Status LoadFrozenModel(tensorflow::Session *sess, string graph_fn) {
  tensorflow::Status status;

  // Read in the protobuf graph we exported
  tensorflow::GraphDef graph_def;
  status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
  if( status != tensorflow::Status::OK() ) return status;

  // Create the graph in the current session
  status = sess->Create(graph_def);
  if( status != tensorflow::Status::OK() ) return status;

  return tensorflow::Status::OK();
}

int main(int argc, char const *argv[]) {

  if( argc < 3 ) {
    cerr << argv[0] << " <frozen_graph_fn> <vocab_fn>" << endl;
    return 1;
  } 

  const string frozen_graph_fn = argv[1];
  const string vocab_fn = argv[2];

  // Prepare session
  tensorflow::Session *sess;
  tensorflow::SessionOptions options;
  TF_CHECK_OK(tensorflow::NewSession(options, &sess));
  LoadLSTMLibrary();
  TF_CHECK_OK(LoadFrozenModel(sess, frozen_graph_fn));

  // Prepare config, vocab, input
  Config config = Config(300, 15, true); // wrd_dim=300, word_length=15, use_crf=true
  Vocab vocab = Vocab(vocab_fn, false);  // lowercase=false
  // set class_size to config
  config.SetClassSize(vocab.GetTagVocabSize());
  cerr << "class size = " << config.GetClassSize() << endl;

  vector<string> bucket;
  for( string line; getline(cin, line); ) {
    if( line == "" ) {
       Input input = Input(config, vocab, bucket);
       bucket.clear();
//#ifdef DEBUG
       int max_sentence_length = input.GetMaxSentenceLength();
       cout << "[word ids]" << endl;
       tensorflow::Tensor* sentence_word_ids = input.GetSentenceWordIds();
       auto data_word_ids = sentence_word_ids->flat<float>().data();
       for( int i = 0; i < max_sentence_length; i++ ) {
         cout << data_word_ids[i] << " ";
       }
       cout << endl;
       cout << "[wordchr ids]" << endl;
       tensorflow::Tensor* sentence_wordchr_ids = input.GetSentenceWordChrIds();
       auto data_wordchr_ids = sentence_wordchr_ids->flat<float>().data();
       int word_length = config.GetWordLength();
       for( int i = 0; i < max_sentence_length; i++ ) {
         for( int j = 0; j < word_length; j++ ) {
           cout << data_wordchr_ids[i*word_length + j] << " ";
         }
         cout << endl;
       }
       cout << "[pos ids]" << endl;
       tensorflow::Tensor* sentence_pos_ids = input.GetSentencePosIds();
       auto data_pos_ids = sentence_pos_ids->flat<float>().data();
       for( int i = 0; i < max_sentence_length; i++ ) {
         cout << data_pos_ids[i] << " ";
       }
       cout << endl;
       cout << "[etcs]" << endl;
       tensorflow::Tensor* sentence_etcs = input.GetSentenceEtcs();
       auto data_etcs = sentence_etcs->flat<float>().data();
       int etc_dim = config.GetEtcDim();
       for( int i = 0; i < max_sentence_length; i++ ) {
         for( int j = 0; j < etc_dim; j++ ) {
           cout << data_etcs[i*etc_dim + j] << " ";
         }
         cout << endl;
       }

       cout << endl;
//#endif
    } else {
       bucket.push_back(line);
    }
  }

  sess->Close();
  return 0;

  // Prepare inputs
  tensorflow::TensorShape data_shape({1, 4});
  tensorflow::Tensor data(tensorflow::DT_FLOAT, data_shape);
  auto data_ = data.flat<float>().data();
  data_[0] = 2;
  data_[1] = 14;
  data_[2] = 33;
  data_[3] = 50;
  tensor_dict feed_dict = {
      {"X", data},
  };

  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(sess->Run(feed_dict, {"logits"},
                        {}, &outputs));

  std::cout << "input           " << data.DebugString() << std::endl;
  std::cout << "logits          " << outputs[0].DebugString() << std::endl;

  tensorflow::TTypes<float>::Flat logits_flat = outputs[0].flat<float>();
  for( int i = 0; i < 3; i++ ) {
    const float logit = logits_flat(i);
    std::cout << "logit           " << i << "," << logit << std::endl; 
  } 

  return 0;
}
