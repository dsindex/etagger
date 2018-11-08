#include "Input.h"
#include "TFUtil.h"

#include <cstdio>
#include <sys/time.h>

int main(int argc, char const *argv[])
{
  if( argc < 3 ) {
    cerr << argv[0] << " <frozen_graph_fn> <vocab_fn>" << endl;
    return 1;
  } 

  const string frozen_graph_fn = argv[1];
  const string vocab_fn = argv[2];

  TFUtil util = TFUtil();
  tensorflow::Session* sess = util.CreateSession();
  TF_CHECK_OK(util.LoadFrozenModel(sess, frozen_graph_fn));

  Config config = Config(15, true); // word_length=15, use_crf=true
  Vocab vocab = Vocab(vocab_fn, true);  // lowercase=true
  config.SetClassSize(vocab.GetTagVocabSize());
  cerr << "class size = " << config.GetClassSize() << endl;

  struct timeval t1,t2,t3,t4;
  int num_buckets = 0;
  double total_duration_time = 0.0;
  gettimeofday(&t1, NULL);

  vector<string> bucket;
  for( string line; getline(cin, line); ) {
    if( line == "" ) {
       gettimeofday(&t3, NULL);

       Input input = Input(config, vocab, bucket);
       int max_sentence_length = input.GetMaxSentenceLength();
       tensorflow::Tensor* sentence_word_ids = input.GetSentenceWordIds();
       tensorflow::Tensor* sentence_wordchr_ids = input.GetSentenceWordChrIds();
       tensorflow::Tensor* sentence_pos_ids = input.GetSentencePosIds();
       tensorflow::Tensor* sentence_etcs = input.GetSentenceEtcs();
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
       cout << "[etcs]" << endl;
       auto data_etcs = sentence_etcs->flat<float>().data();
       int etc_dim = config.GetEtcDim();
       for( int i = 0; i < max_sentence_length; i++ ) {
         for( int j = 0; j < etc_dim; j++ ) {
           cout << data_etcs[i*etc_dim + j] << " ";
         }
         cout << endl;
       }
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
         {"input_data_etcs", *sentence_etcs},
         {"sentence_length", *sentence_length},
         {"is_train", *is_train},
       };
       std::vector<tensorflow::Tensor> outputs;
       TF_CHECK_OK(sess->Run(feed_dict, {"logits", "loss/trans_params"},
                        {}, &outputs));
       /*
       cout << "logits           " << outputs[0].DebugString() << endl;
       cout << "trans_params     " << outputs[1].DebugString() << endl;
       */
       int class_size = config.GetClassSize();
       tensorflow::TTypes<float>::Flat logits_flat = outputs[0].flat<float>();
       tensorflow::TTypes<float>::Flat trans_params_flat = outputs[1].flat<float>();
       if( config.GetUseCRF() ) {
         vector<int> viterbi(max_sentence_length);
         util.ViterbiDecode(logits_flat, trans_params_flat, max_sentence_length, class_size, viterbi);
         for( int i = 0; i < max_sentence_length; i++ ) {
           int max_j = viterbi[i];
           string tag = vocab.GetTag(max_j);
           cout << bucket[i] + " " + tag << endl;
         }
         cout << endl;
       } else {
         for( int i = 0; i < max_sentence_length; i++ ) {
           int max_j = 0;
           float max_logit = numeric_limits<float>::min();
           for( int j = 0; j < class_size; j++ ) {
             const float logit = logits_flat(i*class_size + j);
             if( logit > max_logit ) {
               max_logit = logit;
               max_j = j;
             }
           }
           string tag = vocab.GetTag(max_j);
           cout << bucket[i] + " " + tag << endl;
         }
         cout << endl;
       }

       num_buckets += 1;
       bucket.clear();

       gettimeofday(&t4, NULL);
       double duration_time = ((t4.tv_sec - t3.tv_sec)*1000000 + t4.tv_usec - t3.tv_usec)/(double)1000000;
       fprintf(stderr,"elapsed time per sentence = %lf sec\n", duration_time);
       total_duration_time += duration_time;
    } else {
       bucket.push_back(line);
    }
  }
  gettimeofday(&t2, NULL);
  double duration_time = ((t2.tv_sec - t1.tv_sec)*1000000 + t2.tv_usec - t1.tv_usec)/(double)1000000;
  fprintf(stderr,"elapsed time = %lf sec\n", duration_time);
  fprintf(stderr,"duration time on average = %lf sec\n", total_duration_time / num_buckets);

  util.DestroySession(sess);
  return 0;
}
