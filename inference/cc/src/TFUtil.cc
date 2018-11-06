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
  // Load _lstm_ops.so library(from LB_LIBRARY_PATH) for LSTMBlockFusedCell()
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

TFUtil::~TFUtil()
{
}
