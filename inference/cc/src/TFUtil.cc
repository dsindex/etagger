#include "TFUtil.h"

/*
 *  public methods
 */

TFUtil::TFUtil()
{
}

tensorflow::Session* TFUtil::CreateSession()
{
  tensorflow::Session* sess;
  tensorflow::SessionOptions options;
  tensorflow::ConfigProto& conf = options.config;
  conf.set_inter_op_parallelism_threads(1);
  conf.set_intra_op_parallelism_threads(1);
  TF_CHECK_OK(tensorflow::NewSession(options, &sess));
  return sess;
}

void TFUtil::DestroySession(tensorflow::Session* sess)
{
  if( sess ) sess->Close();
}

void TFUtil::LoadLSTMLibrary()
{
  /*
   *  Load _lstm_ops.so library(from LB_LIBRARY_PATH) for LSTMBlockFusedCell()
   */
  TF_Status* status = TF_NewStatus();
  TF_LoadLibrary("_lstm_ops.so", status);
  if( TF_GetCode(status) != TF_OK ) {
    throw runtime_error("fail to load _lstm_ops.so");
  }
  TF_DeleteStatus(status);
}

tensorflow::Status TFUtil::LoadFrozenModel(tensorflow::Session* sess, string graph_fn)
{
  tensorflow::Status status;

  LoadLSTMLibrary();

  // Read in the protobuf graph we freezed
  tensorflow::GraphDef graph_def;
  status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
  if( status != tensorflow::Status::OK() ) return status;

  // Create the graph in the current session
  status = sess->Create(graph_def);
  if( status != tensorflow::Status::OK() ) return status;

  return tensorflow::Status::OK();
}

tensorflow::Status TFUtil::LoadModel(tensorflow::Session *sess,
                                     string graph_fn,
                                     string checkpoint_fn = "")
{

  /* 
   *  source is from https://github.com/PatWie/tensorflow-cmake/blob/master/inference/cc/inference_cc.cc 
   */
  tensorflow::Status status;

  // Read in the protobuf graph we exported
  tensorflow::MetaGraphDef graph_def;
  status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
  if (status != tensorflow::Status::OK()) return status;

  // create the graph in the current session
  status = sess->Create(graph_def.graph_def());
  if (status != tensorflow::Status::OK()) return status;

  // restore model from checkpoint, iff checkpoint is given
  if (checkpoint_fn != "") {
    const string restore_op_name = graph_def.saver_def().restore_op_name();
    const string filename_tensor_name =
        graph_def.saver_def().filename_tensor_name();

    tensorflow::Tensor filename_tensor(tensorflow::DT_STRING,
                                       tensorflow::TensorShape());
    filename_tensor.scalar<string>()() = checkpoint_fn;

    tensor_dict feed_dict = {{filename_tensor_name, filename_tensor}};
    status = sess->Run(feed_dict, {}, {restore_op_name}, nullptr);
    if (status != tensorflow::Status::OK()) return status;
  } else {
    // virtual Status Run(const vector<pair<string, Tensor> >& inputs,
    //                  const vector<string>& output_tensor_names,
    //                  const vector<string>& target_node_names,
    //                  vector<Tensor>* outputs) = 0;
    status = sess->Run({}, {}, {"init"}, nullptr);
    if (status != tensorflow::Status::OK()) return status;
  }

  return tensorflow::Status::OK();
}

TFUtil::~TFUtil()
{
}
