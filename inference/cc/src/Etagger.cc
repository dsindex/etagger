#include "Etagger.h"

/*
 *  public methods
 */

Etagger::Etagger(string frozen_graph_fn, string vocab_fn, int word_length, bool lowercase, bool is_memmapped, int num_threads)
{
  /*
   *  Args:
   *    frozen_graph_fn: path to a file of frozen graph.
   *    vocab_fn: path to a vocab file.
   *    word_length: max character size of word. ex) 15
   *    lowercase: true if vocab file was all lowercased. ex) true
   *    is_memmapped: true if frozen graph was memmapped, otherwise false.
   *    num_threads: number of threads for tensorflow. 0 for all cores, n for n cores.
   */ 
  this->util = new TFUtil();
  this->sess = NULL;

  if( is_memmapped ) {
    tensorflow::MemmappedEnv* memmapped_env = this->util->CreateMemmappedEnv(frozen_graph_fn); 
    this->sess = this->util->CreateSession(memmapped_env, num_threads);
    TF_CHECK_OK(this->util->LoadFrozenMemmappedModel(memmapped_env, this->sess));
  } else {
    this->sess = this->util->CreateSession(NULL, num_threads);
    TF_CHECK_OK(this->util->LoadFrozenModel(this->sess, frozen_graph_fn));
  }
 
  this->config = new Config(word_length);
  this->vocab = new Vocab(vocab_fn, lowercase);
  this->config->SetClassSize(this->vocab->GetTagVocabSize());
  cerr << "class size = " << this->config->GetClassSize() << endl;
}

int Etagger::Analyze(vector<string>& bucket)
{
  /*
   *  Args:
   *    bucket: list of 'word pos chk tag'
   *  Returns:
   *    number of tokens.
   *    -1 if failed.
   *    analyzed results are saved to bucket itself.
   *    bucket: list of 'word pos chk tag predict'
   */
  Input input = Input(this->config, this->vocab, bucket);
  int max_sentence_length = input.GetMaxSentenceLength();
  tensorflow::Tensor* sentence_word_ids = input.GetSentenceWordIds();
  tensorflow::Tensor* sentence_wordchr_ids = input.GetSentenceWordChrIds();
  tensorflow::Tensor* sentence_pos_ids = input.GetSentencePosIds();
  tensorflow::Tensor* sentence_chk_ids = input.GetSentenceChkIds();
  tensorflow::Tensor* sentence_length = input.GetSentenceLength();
  tensorflow::Tensor* is_train = input.GetIsTrain();
#ifdef DEBUG
  cout << "[word ids]" << endl;
  auto data_word_ids = sentence_word_ids->flat<int>().data();
  for( int i = 0; i < max_sentence_length; i++ ) {
    cout << data_word_ids[i] << " ";
  }
  cout << endl;
  cout << "[wordchr ids]" << endl;
  auto data_wordchr_ids = sentence_wordchr_ids->flat<int>().data();
  int word_length = config.GetWordLength();
  for( int i = 0; i < max_sentence_length; i++ ) {
    for( int j = 0; j < word_length; j++ ) {
      cout << data_wordchr_ids[i*word_length + j] << " ";
    }
    cout << endl;
  }
  cout << "[pos ids]" << endl;
  auto data_pos_ids = sentence_pos_ids->flat<int>().data();
  for( int i = 0; i < max_sentence_length; i++ ) {
    cout << data_pos_ids[i] << " ";
  }
  cout << endl;
  cout << "[chk ids]" << endl;
  auto data_chk_ids = sentence_chk_ids->flat<int>().data();
  for( int i = 0; i < max_sentence_length; i++ ) {
    cout << data_chk_ids[i] << " ";
  }
  cout << endl;
  cout << "[sentence length]" << endl;
  auto data_sentence_length = sentence_length->flat<int>().data();
  cout << *data_sentence_length << endl;
  cout << "[is_train]" << endl;
  auto data_is_train = is_train->flat<bool>().data();
  cout << *data_is_train << endl;

  cout << endl;
#endif
  tensor_dict feed_dict = {
    {"input_data_word_ids", *sentence_word_ids},
    {"input_data_wordchr_ids", *sentence_wordchr_ids},
    {"input_data_pos_ids", *sentence_pos_ids},
    {"input_data_chk_ids", *sentence_chk_ids},
    {"sentence_length", *sentence_length},
    {"is_train", *is_train},
  };
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status run_status = this->sess->Run(feed_dict, {"logits_indices"}, {}, &outputs);
  if( !run_status.ok() ) {
    cerr << run_status.error_message() << endl;
    return -1;
  }
  /*
  cout << "logits_indices   " << outputs[0].DebugString() << endl;
  */
  int class_size = this->config->GetClassSize();
  tensorflow::TTypes<int>::Flat logits_indices_flat = outputs[0].flat<int>();
  for( int i = 0; i < max_sentence_length; i++ ) {
    int max_idx = logits_indices_flat(i);
    string tag = this->vocab->GetTag(max_idx);
    bucket[i] = bucket[i] + " " + tag;
  }
  return max_sentence_length;
}

Etagger::~Etagger()
{
  delete this->vocab;
  delete this->config;
  this->util->DestroySession(this->sess);
  delete this->util;
}

