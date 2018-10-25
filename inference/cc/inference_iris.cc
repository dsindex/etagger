// source template is from https://github.com/PatWie/tensorflow-cmake/blob/master/inference/cc/inference_cc.cc
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <iostream>
#include <string>

typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;

tensorflow::Status LoadModel(tensorflow::Session *sess, std::string graph_fn,
                             std::string checkpoint_fn = "") {
  tensorflow::Status status;

  // Read in the protobuf graph we exported
  tensorflow::MetaGraphDef graph_def;
  status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
  if (status != tensorflow::Status::OK()) return status;

  // create the graph in the current session
  status = sess->Create(graph_def.graph_def());
  if (status != tensorflow::Status::OK()) return status;

  // initialize all variables
  tensor_dict feed_dict_init = {}; 
  status = sess->Run(feed_dict_init, {}, {"init_all_vars_op"}, nullptr);
  if (status != tensorflow::Status::OK()) return status;

  // restore model from checkpoint, iff checkpoint is given
  if (checkpoint_fn != "") {
    const std::string restore_op_name = graph_def.saver_def().restore_op_name();
    const std::string filename_tensor_name =
        graph_def.saver_def().filename_tensor_name();

    tensorflow::Tensor filename_tensor(tensorflow::DT_STRING,
                                       tensorflow::TensorShape());
    filename_tensor.scalar<std::string>()() = checkpoint_fn;

    tensor_dict feed_dict = {{filename_tensor_name, filename_tensor}};
    status = sess->Run(feed_dict, {}, {restore_op_name}, nullptr);
    if (status != tensorflow::Status::OK()) return status;
  } else {
    // virtual Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
    //                  const std::vector<string>& output_tensor_names,
    //                  const std::vector<string>& target_node_names,
    //                  std::vector<Tensor>* outputs) = 0;
    status = sess->Run({}, {}, {"init"}, nullptr);
    if (status != tensorflow::Status::OK()) return status;
  }

  return tensorflow::Status::OK();
}

int main(int argc, char const *argv[]) {
  const std::string graph_fn = "./exported/iris_model.meta";
  const std::string checkpoint_fn = "./exported/iris_model";

  // prepare session
  tensorflow::Session *sess;
  tensorflow::SessionOptions options;
  TF_CHECK_OK(tensorflow::NewSession(options, &sess));
  TF_CHECK_OK(LoadModel(sess, graph_fn, checkpoint_fn));

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
