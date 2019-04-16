#include "Etagger.h"

#include <cstdio>
#include <sys/time.h>

int main(int argc, char const *argv[])
{
  if( argc < 3 ) {
    cerr << argv[0] << " <frozen_graph_fn> <vocab_fn> [is_memmapped(1 | 0:default)]" << endl;
    return 1;
  } 

  const string frozen_graph_fn = argv[1];
  const string vocab_fn = argv[2];
  bool is_memmapped = false;
  if( argc == 4 && argv[3][0] == '1' ) is_memmapped = true;

  Etagger etagger = Etagger(frozen_graph_fn,
                            vocab_fn,
                            15,   // word_length = 15
                            true, // lowercase = true
                            is_memmapped,
                            0);   // 0(all cores) | n(n cores)

  struct timeval t1,t2,t3,t4;
  int num_buckets = 0;
  double total_duration_time = 0.0;
  gettimeofday(&t1, NULL);

  vector<string> bucket;
  for( string line; getline(cin, line); ) {
    if( line == "" ) {
       gettimeofday(&t3, NULL);

       int ret = etagger.Analyze(bucket);
       if( ret < 0 ) continue;
       for( int i = 0; i < ret; i++ ) {
         cout << bucket[i] << endl;
       }
       cout << endl;

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

  return 0;
}
