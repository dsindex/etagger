#include "TFUtil.h"

/*
 *  public methods
 */

TFUtil::TFUtil()
{
}

tensorflow::MemmappedEnv* TFUtil::CreateMemmappedEnv(string graph_fn)
{
  tensorflow::MemmappedEnv* memmapped_env = new tensorflow::MemmappedEnv(tensorflow::Env::Default());
  TF_CHECK_OK(memmapped_env->InitializeFromFile(graph_fn));
  return memmapped_env;
}

tensorflow::Session* TFUtil::CreateSession(tensorflow::MemmappedEnv* memmapped_env, int num_threads = 0)
{
  tensorflow::Session* sess;
  tensorflow::SessionOptions options;

  if( memmapped_env ) {
    options.config.mutable_graph_options()->mutable_optimizer_options()->set_opt_level(::tensorflow::OptimizerOptions::L0);
    options.env = memmapped_env;
  }

  tensorflow::ConfigProto& conf = options.config;
  if( num_threads > 0 ) {
    conf.set_inter_op_parallelism_threads(num_threads);
    conf.set_intra_op_parallelism_threads(num_threads);
  }
  TF_CHECK_OK(tensorflow::NewSession(options, &sess));
  return sess;
}

void TFUtil::DestroySession(tensorflow::Session* sess)
{
  if( sess ) sess->Close();
}

tensorflow::Status TFUtil::LoadFrozenModel(tensorflow::Session* sess, string graph_fn)
{
  tensorflow::Status status;

  load_lstm_lib();
  /* load_qrnn_lib(); */

  // Read in the protobuf graph freezed
  tensorflow::GraphDef graph_def;
  status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
  if( status != tensorflow::Status::OK() ) return status;

  // Create the graph in the current session
  status = sess->Create(graph_def);
  if( status != tensorflow::Status::OK() ) return status;

  return tensorflow::Status::OK();
}

tensorflow::Status TFUtil::LoadFrozenMemmappedModel(tensorflow::MemmappedEnv* memmapped_env, tensorflow::Session* sess, string graph_fn)
{
  tensorflow::Status status;

  load_lstm_lib();
  /* load_qrnn_lib(); */

  // Read the memmory-mapped graph
  tensorflow::GraphDef graph_def;
  status = ReadBinaryProto(memmapped_env, tensorflow::MemmappedFileSystem::kMemmappedPackageDefaultGraphDef, &graph_def);

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

/*
 *  private methods
 */

void TFUtil::load_lstm_lib()
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

void TFUtil::load_qrnn_lib()
{
  /*
   *  Load qrnn_lib.cpython-36m-x86_64-linux-gnu.so library(from LB_LIBRARY_PATH) for QRNN
   */
  TF_Status* status = TF_NewStatus();
  TF_LoadLibrary("qrnn_lib.cpython-36m-x86_64-linux-gnu.so", status);
  if( TF_GetCode(status) != TF_OK ) {
    throw runtime_error("fail to load qrnn_lib.cpython-36m-x86_64-linux-gnu.so");
  }
  TF_DeleteStatus(status);
}
