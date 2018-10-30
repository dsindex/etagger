#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <iostream>
#include <string>

typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;

tensorflow::Status LoadFrozenModel(tensorflow::Session *sess, std::string graph_fn) {
  tensorflow::Status status;

  // Read in the protobuf graph we exported
  tensorflow::GraphDef graph_def;
  status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
  if (status != tensorflow::Status::OK()) return status;

  // create the graph in the current session
  status = sess->Create(graph_def);
  if (status != tensorflow::Status::OK()) return status;

  return tensorflow::Status::OK();
}

int main(int argc, char const *argv[]) {

  const std::string graph_fn = "./exported/iris_frozen.pb";

  // prepare session
  tensorflow::Session *sess;
  tensorflow::SessionOptions options;
  TF_CHECK_OK(tensorflow::NewSession(options, &sess));
  TF_CHECK_OK(LoadFrozenModel(sess, graph_fn));

  // prepare inputs
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

  return 0;
}
