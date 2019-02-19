#include "TFUtil.h"

/*
 *  public methods
 */

TFUtil::TFUtil()
{
}

tensorflow::Session* TFUtil::CreateSession(int num_threads = 0)
{
  tensorflow::Session* sess;
  tensorflow::SessionOptions options;
  tensorflow::ConfigProto& conf = options.config;
  if( num_threads > 0 ) {
    conf.set_inter_op_parallelism_threads(num_threads);
    conf.set_intra_op_parallelism_threads(num_threads);
  }
  TF_CHECK_OK(tensorflow::NewSession(options, &sess));
  return sess;
}

tensorflow::MemmappedEnv* TFUtil::CreateMemmappedEnv(string graph_fn)
{
  tensorflow::MemmappedEnv* memmapped_env = new tensorflow::MemmappedEnv(tensorflow::Env::Default());
  TF_CHECK_OK(memmapped_env->InitializeFromFile(graph_fn));
  return memmapped_env;
}

tensorflow::Session* TFUtil::CreateMemmappedEnvSession(tensorflow::MemmappedEnv* memmapped_env, int num_threads = 0)
{
  tensorflow::Session* sess;
  tensorflow::SessionOptions options;

  options.config.mutable_graph_options()->mutable_optimizer_options()->set_opt_level(::tensorflow::OptimizerOptions::L0);
  options.env = memmapped_env;

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

void TFUtil::ViterbiDecode(tensorflow::TTypes<float>::Flat logits_flat,
                           tensorflow::TTypes<float>::Flat trans_params_flat,
                           int max_sentence_length,
                           int class_size,
                           vector<int>& viterbi)
{
  /*
   *  Args:
   *    logits_flat: A [max_sentence_length, class_size] matrix virtually
   *    trans_params_flat: A [class_size, class_size] matrix virtually
   *    max_sentence_length: integer, sentence length
   *    class_size: integer, number of classes(or tags)
   *
   *  Returns:
   *    viterbi: A [max_sentence_length] list of integers containing the highest scoring class indices. 
   */

  vector<vector<tuple<float,int>>> lattice(max_sentence_length+1, vector<tuple<float,int>>(class_size, make_tuple(0.0, -1)));

  for( int i = 0; i < max_sentence_length; i++ ) {
    if( i == 0 ) {
      // initialize
      for( int j = 0; j < class_size; j++ ) {
        float logit = logits_flat(i*class_size + j); // (0, j)
        float weight = 0.0 + logit;
        get<0>(lattice[i][j]) = weight; // path prob
        get<1>(lattice[i][j]) = -1;     // path node
      }
    }
    for( int j = 0; j < class_size; j++ ) {   // current
      for( int k = 0; k < class_size; k++ ) { // next
        float trans_param = trans_params_flat(j*class_size + k); // j -> k
        float logit = 0.0;
        if( i < max_sentence_length ) logit = logits_flat((i+1)*class_size + k); // (i+1, k)
        float weight = trans_param + logit;
        if( get<1>(lattice[i+1][k]) == -1 ) { // first
          get<0>(lattice[i+1][k]) = weight + get<0>(lattice[i][j]);
          get<1>(lattice[i+1][k]) = j;
        } else { // update
          if( get<0>(lattice[i+1][k]) < weight + get<0>(lattice[i][j]) ) {
            get<0>(lattice[i+1][k]) = weight + get<0>(lattice[i][j]);
            get<1>(lattice[i+1][k]) = j;
          }
        }
      }
    }
  }

#ifdef DEBUG
  for( int i = 0; i < max_sentence_length+1; i++ ) {
    for( int j = 0; j < class_size; j++ ) {
      float path_prob = get<0>(lattice[i][j]);
      int   path_node = get<1>(lattice[i][j]);
      cout << "(" << i << ", " << j << ") => " << path_prob << ", " << path_node << endl;
    }
  }
#endif

  // find max index at the last
  int max_j = 0;
  float max_prob = 0.0;
  for( int j = 0; j < class_size; j++ ) {
    if( get<0>(lattice[max_sentence_length][j]) > max_prob ) {
      max_prob = get<0>(lattice[max_sentence_length][j]);
      max_j = j;
    }
  }

  // back-tracking
  int j = max_j;
  for( int i = max_sentence_length; i >= 0; i-- ) {
    j = get<1>(lattice[i][j]);
    if( i != 0 ) {
      viterbi[i-1] = j;
    }
  }
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
