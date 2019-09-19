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
   *    lowercase: true if vocab file was all lowercased, otherwise false.
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
  cerr << "Loading graph and creating session ... done" << endl; 
 
  this->config = new Config(word_length);
  cerr << "Loading Config ... done" << endl; 
  cerr << "Loading Vocab From " << vocab_fn;
  this->vocab = new Vocab(vocab_fn, lowercase);
  cerr << " ... done" << endl;
  this->config->SetClassSize(this->vocab->GetTagVocabSize());
  cerr << "Class size: " << this->config->GetClassSize() << endl;
}

int Etagger::Analyze(vector<string>& bucket)
{
  /*
   *  Args:
   *    bucket: list of 'word pos chk tag'
   *
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
  int word_length = this->config->GetWordLength();
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

/*
 *  public methods for C
 */

extern "C" {

  Etagger* initialize(const char* frozen_graph_fn,
                      const char* vocab_fn,
                      int word_length,
                      int lowercase,
                      int is_memmapped,
                      int num_threads)
  {
    /*
     *  Args:
     *    frozen_graph_fn: path to a file of frozen graph.
     *    vocab_fn: path to a vocab file.
     *    word_length: max character size of word. ex) 15
     *    lowercase: 1 if vocab file was all lowercased, otherwise 0.
     *    is_memmapped: 1 if frozen graph was memmapped, otherwise 0.
     *    num_threads: number of threads for tensorflow. 0 for all cores, n for n cores.
     *
     *  Python: 
     *    import ctypes as c
     *    so_path = 'path-to/lib' + '/' + 'libetagger.so'    
     *    libetagger = c.cdll.LoadLibrary(so_path)
     *
     *    frozen_graph_fn = c.c_char_p(b'path-to/ner_frozen.pb')
     *    vocab_fn = c.c_char_p(b'path-to/vocab.txt')
     *    word_length = c.c_int(15)
     *    lowercase = c.c_int(1)
     *    is_memmapped = c.c_int(1)
     *    num_threads = c.c_int(0)
     *    etagger = libetagger.initialize(frozen_graph_fn, vocab_fn, word_length, lowercase, is_memmapped, num_threads)
     */ 
    bool b_lowercase = false;
    if( lowercase ) b_lowercase = true;
    bool b_is_memmapped = false;
    if( is_memmapped ) b_is_memmapped = true;
    return new Etagger(frozen_graph_fn, vocab_fn, word_length, b_lowercase, b_is_memmapped, num_threads);
  } 

  static void split(string s, vector<string>& tokens)
  {
    istringstream iss(s);
    for( string ts; iss >> ts; )
      tokens.push_back(ts);
  }

  int analyze(Etagger* etagger, struct result_obj* robj, int max)
  {
    /*
     *  Args:
     *    etagger: an instance of Etagger , i.e, handler.
     *    robj: list of result_obj.
     *    max:  max size of robj.
     *
     *  Python:
     *    class Result( c.Structure ):
     *        _fields_ = [('word', c.c_char * MAX_WORD ),
     *                    ('pos', c.c_char * MAX_POS ),
     *                    ('chk', c.c_char * MAX_CHK ),
     *                    ('tag', c.c_char * MAX_TAG ),
     *                    ('predict', c.c_char * MAX_TAG )]
     *
     *    bucket = build_bucket(nlp, line)
     *    # ex) bucket
     *    #     word    pos chk tag
     *    #     ...
     *    #     jeju    NNP O   B-GPE
     *    #     island  NN  O   O
     *    #     ...
     *    max_sentence_length = len(bucket)
     *    robj = (Result * max_sentence_length)()
     *    # fill robj from bucket.
     *    for i in range(max_sentence_length):
     *        tokens = bucket[i].split()
     *        robj[i].word = tokens[0].encode('utf-8')
     *        robj[i].pos = tokens[1].encode('utf-8')
     *        robj[i].chk = tokens[2].encode('utf-8')
     *        robj[i].tag = tokens[3].encode('utf-8')
     *        robj[i].predict = b'O'
     *    c_max_sentence_length = c.c_int(max_sentence_length)
     *    ret = libetagger.analyze(etagger, c.byref(robj), c_max_sentence_length)
     *    out = []
     *    for r in robj:
     *        out.append([r.word.decode('utf-8'),
     *        r.pos.decode('utf-8'),
     *        r.chk.decode('utf-8'),
     *        r.tag.decode('utf-8'),
     *        r.predict.decode('utf-8')])
     *
     *  Returns:
     *    number of tokens.
     *    -1 if failed.   
     *    analyzed results are saved to robj itself.
     */
    vector<string> bucket;

    // build bucket from robj
    for( int i = 0; i < max; i++ ) {
      string s = string(robj[i].word) + " " + 
                 string(robj[i].pos) + " " + 
                 string(robj[i].chk) + " " + 
                 string(robj[i].tag);
      bucket.push_back(s);
    }
    // bucket: list of 'word pos chk tag'

    int ret = etagger->Analyze(bucket);
    if( ret < 0 ) return -1;
    // bucket: list of 'word pos chk tag predict'

    // assign predict to robj
    for( int i = 0; i < max; i++ ) {
      vector<string> tokens;
      split(bucket[i], tokens);
      string predict = tokens[4]; // last one
      strncpy(robj[i].predict, predict.c_str(), MAX_TAG);
    }

    return ret;
  }

  void finalize(Etagger* etagger)
  {
    /*
     *  Args:
     *    etagger: an instance of Etagger , handler
     *  Python:
     *    libetagger.finalize(etagger)
     */
    if( etagger ) {
      delete etagger;
    }
  }

}
