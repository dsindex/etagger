
### History ( ~ 2020. 2. 25)

- Evaluation script
  -  etc/token_eval.py
  - etc/chunk_eval.py
  - etc/conlleval
- The bellow results for BERT is not valid now. because BERT is used as feature-based currently.
  - checkout the code for BERT fine-tuning: https://github.com/dsindex/etagger/tree/7354971552bbf204a4357369637b687c1704bdcc

- summary
  - https://docs.google.com/spreadsheets/d/1Sy7YREtqsIaaesNM1LAdsstomB7nrQV483XPzwH5JBM/edit?usp=sharing

----

- experiments 1 data
```
1. number of labels
610

2. data
10M,  81563 sentences, data/kor.train.txt
1.7M, 13632 sentences, data/kor.dev.txt
2.4M, 18937 sentences, data/kor.test.txt
375K, 3019 sentences, data/kor.test.confirmed.txt

3. glove
2.5G(500k, 525470) kor.glove.300d.txt
841M(300k, 308383) kor.glove.300k.300d.txt

4. elmo
358M kor_elmo_2x4096_512_2048cnn_2xhighway_622k_weights.hdf5
336  kor_elmo_2x4096_512_2048cnn_2xhighway_622k_options.json

358M kor_elmo_2x4096_512_2048cnn_2xhighway_882k_weights.hdf5
336  kor_elmo_2x4096_512_2048cnn_2xhighway_882k_options.json

358M kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5
336  kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json

5. bert

1) all.dha.2.5m_step
  768    hidden_size
  202592 vocab_size
  2.5G   bert_model.ckpt
  64     bert_max_seq_length (55)

2) multi_cased_L-12_H-768_A-12
  768    hidden_size
  119547 vocab_size
  682M   bert_model.ckpt
  96     bert_max_seq_length (88)

3) all.200k.out.1m-step.reduced
  768    hidden_size
  100795 vocab_size
  627M   bert_model.ckpt
  64     bert_max_seq_length (55)

4) all.bpe.4.8m_step
  768    hidden_size
  100102 vocab.txt
  1.9G   bert_model.ckpt
  64     bert_max_seq_length (61)

5. evaluation by CRF(wapiti)
token : 0.900468066343
chunk : 0.9141800490902886 
conlleval : 91.42

```

- how to run
```
- embedding
* for Glove
$ python embvec.py --emb_path embeddings/kor.glove.300k.300d.txt --wrd_dim 300 --train_path data/kor.train.txt > embeddings/kor_vocab.txt

* for ELMo
$ python embvec.py --emb_path embeddings/kor.glove.300k.300d.txt --wrd_dim 300 --train_path data/kor.train.txt --elmo_vocab_path embeddings/kor_elmo_vocab.txt --elmo_options_path embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_622k_options.json --elmo_weight_path embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_622k_weights.hdf5 > embeddings/kor_vocab.txt
$ python embvec.py --emb_path embeddings/kor.glove.300k.300d.txt --wrd_dim 300 --train_path data/kor.train.txt --elmo_vocab_path embeddings/kor_elmo_vocab.txt --elmo_options_path embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_882k_options.json --elmo_weight_path embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_882k_weights.hdf5 > embeddings/kor_vocab.txt
$ python embvec.py --emb_path embeddings/kor.glove.300k.300d.txt --wrd_dim 300 --train_path data/kor.train.txt --elmo_vocab_path embeddings/kor_elmo_vocab.txt --elmo_options_path embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json --elmo_weight_path embeddings/kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_weights.hdf5 > embeddings/kor_vocab.txt

* for BERT(all.dha.2.5m_step)
$ python embvec.py --emb_path embeddings/kor.glove.300k.300d.txt --wrd_dim 300 --train_path data/kor.train.txt --bert_config_path embeddings/all.dha.2.5m_step/bert_config.json --bert_vocab_path embeddings/all.dha.2.5m_step/vocab.txt --bert_do_lower_case False --bert_init_checkpoint embeddings/all.dha.2.5m_step/bert_model.ckpt --bert_max_seq_length 64 --bert_dim 768 > embeddings/kor_vocab.txt

* for BERT(multi_cased_L-12_H-768_A-12)
$ python embvec.py --emb_path embeddings/kor.glove.300k.300d.txt --wrd_dim 300 --train_path data/kor.train.txt --bert_config_path embeddings/multi_cased_L-12_H-768_A-12/bert_config.json --bert_vocab_path embeddings/multi_cased_L-12_H-768_A-12/vocab.txt --bert_do_lower_case False --bert_init_checkpoint embeddings/multi_cased_L-12_H-768_A-12/bert_model.ckpt --bert_max_seq_length 96 --bert_dim 768 > embeddings/kor_vocab.txt

* for BERT(all.200k.out.1m-step.reduced)
$ python embvec.py --emb_path embeddings/kor.glove.300k.300d.txt --wrd_dim 300 --train_path data/kor.train.txt --bert_config_path embeddings/all.200k.out.1m-step.reduced/bert_config.json --bert_vocab_path embeddings/all.200k.out.1m-step.reduced/vocab.txt --bert_do_lower_case False --bert_init_checkpoint embeddings/all.200k.out.1m-step.reduced/bert_model.ckpt --bert_max_seq_length 64 --bert_dim 768 > embeddings/vocab.txt

* for BERT(all.bpe.4.8m_step)
$ python embvec.py --emb_path embeddings/kor.glove.300k.300d.txt --wrd_dim 300 --train_path data/kor.train.txt --bert_config_path embeddings/all.bpe.4.8m_step/bert_config.json --bert_vocab_path embeddings/all.bpe.4.8m_step/vocab.txt --bert_do_lower_case False --bert_init_checkpoint embeddings/all.bpe.4.8m_step/bert_model.ckpt --bert_max_seq_length 64 --bert_dim 768 > embeddings/kor_vocab.txt

- train
$ python train.py --emb_path embeddings/kor.glove.300k.300d.txt.pkl --wrd_dim 300 --batch_size 20 --epoch 70

- inference
$ python inference.py --mode bucket --emb_path embeddings/kor.glove.300k.300d.txt.pkl --wrd_dim 300 --restore checkpoint/ner_model < data/kor.test.txt > pred.txt
$ python inference.py --mode bucket --emb_path embeddings/kor.glove.300k.300d.txt.pkl --wrd_dim 300 --restore checkpoint/ner_model < data/kor.test.confirmed.txt > pred.txt

- inference using C++
$ cd inference
$ python export.py --restore ../checkpoint/ner_model --export exported/ner_model --export-pb exported
$ python freeze.py --model_dir exported --output_node_names logits_indices,sentence_lengths --frozen_model_name ner_frozen.pb
$ ln -s ../embeddings embeddings
$ python python/inference.py --emb_path embeddings/kor.glove.300k.300d.txt.pkl --wrd_dim 300 --frozen_path exported/ner_frozen.pb < ../data/kor.test.txt > pred.txt
$ ./cc/build/inference exported/ner_frozen.pb embeddings/kor_vocab.txt < ../data/kor.test.txt > pred.txt
$ ${TENSORFLOW_SOURCE_DIR}/bazel-bin/tensorflow/contrib/util/convert_graphdef_memmapped_format --in_graph=exported/ner_frozen.pb --out_graph=exported/ner_frozen.pb.memmapped
$ ./cc/build/inference exported/ner_frozen.pb.memmapped embeddings/kor_vocab.txt 1 < ../data/kor.test.txt > pred.txt
```

- experiments 1-4
```

* test 1 (kor.test.confirmed.txt)
word embedding size : 300(kor.glove.300k.300d.txt)
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
chk embedding size : 10
chk embedding random init : -0.5 ~ 0.5
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
highway_used : False -> True
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF
+
do_shuffle : False -> True

# trial 1
token : 0.9407237936772046
chunk : 0.9461069735951252
conlleval : 94.60
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.01161017620077825

# trial 2 (decay_steps 50000)
token : 0.936575713393043
chunk : 0.9431633206728162
conlleval : 94.30

```

- experiments 1-3
```
* test 4
word embedding size : 300(kor.glove.300k.300d.txt)
elmo embedding params : kor_elmo_2x4096_512_2048cnn_2xhighway_1000k_options.json
elmo embedding size : 1024
elmo_keep_prob : 0.7
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
chk embedding size : 10
chk embedding random init : -0.5 ~ 0.5
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
highway_used : False
rnn_used : True
rnn_type : fused
rnn_size : 200 -> 250
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF
+
do_shuffle : False -> True

# with kor.test.txt
token : 0.9246887566215298
chunk : 0.9328298457077319
conlleval : 93.28          -> Glove + ELMo + CNN + LSTM + CRF best
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.038068484522230356 sec

# with kor.test.confirmed.txt
token : 0.9449015727528383
chunk : 0.9505084745762712
conlleval : 95.04
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.03945065318610187 sec

* test 3
word embedding size : 300(kor.glove.300k.300d.txt)
elmo embedding params : kor_elmo_2x4096_512_2048cnn_2xhighway_882k_options.json
elmo embedding size : 1024
elmo_keep_prob : 0.7
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
chk embedding size : 10
chk embedding random init : -0.5 ~ 0.5
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
highway_used : False
rnn_used : True
rnn_type : fused
rnn_size : 200 -> 250
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF
+
do_shuffle : False -> True

# with kor.test.txt
token : 0.9254371577805253
chunk : 0.9327268789257093
conlleval : 93.27 
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.028036073652058863 sec

# with kor.test.confirmed.txt
token : 0.9468634301507013
chunk : 0.9513469498541087
conlleval : 95.12
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.027500629977857724 sec

* test 2
word embedding size : 300(kor.glove.300k.300d.txt)
elmo embedding params : kor_elmo_2x4096_512_2048cnn_2xhighway_622k_options.json
elmo embedding size : 1024
elmo_keep_prob : 0.7
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
chk embedding size : 10
chk embedding random init : -0.5 ~ 0.5
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
highway_used : False
rnn_used : True
rnn_type : fused
rnn_size : 200 -> 250
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF
+
do_shuffle : False -> True

# with kor.test.txt
token : 0.9245205352867547
chunk : 0.9322457998895135
conlleval : 93.24
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.02585688952741952 sec

# with kor.test.confirmed.txt
token : 0.9448810701087804
chunk : 0.95053530288657
conlleval : 95.05
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.026471980323298713 sec

* test 1
word embedding size : 300(kor.glove.300k.300d.txt)
elmo embedding params : kor_elmo_2x4096_512_2048cnn_2xhighway_622k_options.json
elmo embedding size : 1024
elmo_keep_prob : 0.7
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
chk embedding size : 10
chk embedding random init : -0.5 ~ 0.5
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
highway_used : False -> True
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF
+
do_shuffle : False -> True

# with kor.test.txt
token : 0.9232528418645322
chunk : 0.9313038296950328
conlleval : 93.13
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.02642091485044801 sec

# with kor.test.confirmed.txt
token : 0.9466062405584206
chunk : 0.9503796095444685
conlleval : 95.04
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.027445672323625582 sec


```

- experiments 1-2
```
* test 10
#word embedding size : 300(kor.glove.300k.300d.txt)
bert embedding : all.bpe.4.8m_step
bert_keep_prob : 0.9
keep_prob : 0.9
#chr_conv_type : conv1d
#chracter embedding size : 25
#chracter embedding random init : -1.0 ~ 1.0
#chk embedding size : 10
#chk embedding random init : -0.5 ~ 0.5
#filter_sizes : [3]
#num_filters : 53
#pos embedding size : 7
#pos embedding random init : -0.5 ~ 0.5
highway_used : False
rnn_used : True
rnn_type : fused
rnn_size : 256
rnn_num_layers : 1
learning_rate : exponential_decay(), 5e-5 / 5000 / 0.9 + Warmup 1epoch + AdamWeightDecayOptimizer
gradient clipping : 1.0
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF
+
Shuffle

#  test.txt
token : 0.9244998239111841
chunk : 0.9312579662554819
conlleval : 93.04
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : pass
# test.confirmed.txt
token : 0.9419287756693527
chunk : 0.9470835025037218
conlleval : 94.65
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : pass

* test 9
#word embedding size : 300(kor.glove.300k.300d.txt)
bert embedding : all.200k.out.1m-step.reduced
bert_keep_prob : 0.9
keep_prob : 0.9
#chr_conv_type : conv1d
#chracter embedding size : 25
#chracter embedding random init : -1.0 ~ 1.0
#chk embedding size : 10
#chk embedding random init : -0.5 ~ 0.5
#filter_sizes : [3]
#num_filters : 53
#pos embedding size : 7
#pos embedding random init : -0.5 ~ 0.5
highway_used : False
rnn_used : True
rnn_type : fused
rnn_size : 256
rnn_num_layers : 1
learning_rate : exponential_decay(), 5e-5 / 5000 / 0.9 + Warmup 1epoch + AdamWeightDecayOptimizer
gradient clipping : 1.0
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF
+
Shuffle

# lr 5e-5
##  test.txt
token : 0.9152224196305857
chunk : 0.9186171817750766
conlleval : 91.78
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : pass
## test.confirmed.txt
token : 0.9363073883429522
chunk : 0.9385187690226582
conlleval : 93.78
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : pass

# lr 2e-5
##  test.txt
token : 0.9137387142390949
chunk : 0.9170126718792128
conlleval : 91.57
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : pass
## test.confirmed.txt
token : 0.936919163075645
chunk : 0.9389798403463673
conlleval : 93.74
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : pass

* test 8
word embedding size : 300(kor.glove.300k.300d.txt)
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
chk embedding size : 10
chk embedding random init : -0.5 ~ 0.5
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
highway_used : False -> True
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF
+
do_shuffle : False -> True

token : 0.9154425358835592
chunk : 0.9252426449106785
conlleval : 92.53
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.011179832594569812

* test 7
word embedding size : 300 -> 300(kor.glove.300k.300d.txt)
keep_prob : 0.7 -> 0.9
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
chk embedding size : 10
chk embedding random init : -0.5 ~ 0.5
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
highway_used : False -> True
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF
+
do_shuffle : False -> True

token : 0.9181024706188728
chunk : 0.9272410471748143
conlleval : 92.73
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.010100270700019095 sec 

* test 6
word embedding size : 300 ->300(kor.glove.300k.300d.txt)
bert embedding : all.200k.out.1m-step.reduced
bert_keep_prob : 0.8
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
chk embedding size : 10
chk embedding random init : -0.5 ~ 0.5
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
highway_used : False -> True
rnn_used : True
rnn_type : fused
rnn_size : 200 -> 256
rnn_num_layers : 2
learning_rate : exponential_decay(), 2e-5 / 5000 / 0.9 -> 0.001 / 12000 / 0.9
gradient clipping : 1.5 -> 10
#learning_rate : use optimization.py from bert, 2e-5 / warmup proportion 0.1
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF

token : 0.9194924561136736
chunk : 0.9274476759989128
conlleval : 92.74          -> Glove + BERT + CNN + LSTM + CRF best
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.04104578713463737 sec

* test 5
word embedding size : 300 ->300(kor.glove.300k.300d.txt)
bert embedding : all.200k.out.1m-step.reduced
bert_keep_prob : 0.8
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
chk embedding size : 10
chk embedding random init : -0.5 ~ 0.5
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200 -> 256
rnn_num_layers : 2
learning_rate : exponential_decay(), 2e-5 / 5000 / 0.9
gradient clipping : 1.5
#learning_rate : use optimization.py from bert, 2e-5 / warmup proportion 0.1
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF

token : 0.91138880999716
chunk : 0.9159176635413299
conlleval : 91.47
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.04353734417941683 sec

* test 4
word embedding size : 300 ->300(kor.glove.300k.300d.txt)
bert embedding : multi_cased_L-12_H-768_A-12
bert_keep_prob : 0.8
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
chk embedding size : 10
chk embedding random init : -0.5 ~ 0.5
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200 -> 256
rnn_num_layers : 2
learning_rate : exponential_decay(), 2e-5 / 5000 / 0.9 -> 0.001 / 12000 / 0.9
gradient clipping : 1.5 -> 10
#learning_rate : use optimization.py from bert, 2e-5 / warmup proportion 0.1
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF

token : 0.9156731999730585
chunk : 0.919030048584135
conlleval : 91.66
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.048045960526195176 sec

* test 3
word embedding size : 300 ->300(kor.glove.300k.300d.txt)
bert embedding : all.dha.2.5m_step
bert_keep_prob : 0.8
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
chk embedding size : 10
chk embedding random init : -0.5 ~ 0.5
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200 -> 256
rnn_num_layers : 2
learning_rate : exponential_decay(), 2e-5 / 5000 / 0.9 -> 0.001 / 12000 / 0.9
gradient clipping : 1.5 -> 10
#learning_rate : use optimization.py from bert, 2e-5 / warmup proportion 0.1
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF

token : 0.9178154959148055
chunk : 0.9258471818457934
conlleval : 92.59
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) :  0.02537741956687146 sec

* test 2
word embedding size : 300 -> 300(kor.glove.300k.300d.txt)
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
chk embedding size : 10
chk embedding random init : -0.5 ~ 0.5
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF

# 300d
token : 0.9195406172849909
chunk : 0.9275132677092717
conlleval : 92.75          -> Glove + CNN + LSTM + CRF best
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.014556412826012726 sec

# 200d, shuffle
token : 0.9169116222658322
chunk : 0.925200607243548
conlleval : 92.52
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.014463350388482146 sec

# 300d, shuffle, without CRF
conlleval : 91.39
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.00722679559548805 sec
  - 32 CPU : 0.003329 sec
  - 1 CPU : 0.004814 sec

* test 1
word embedding size : 300
bert embedding : all.dha.2.5m_step
bert_keep_prob : 0.8
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
chk embedding size : 10
chk embedding random init : -0.5 ~ 0.5
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 256
rnn_num_layers : 2
learning_rate : exponential_decay(), 2e-5 / 5000 / 0.9
gradient clipping : 1.5
#learning_rate : use optimization.py from bert, 2e-5 / warmup proportion 0.1
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF

token : 0.9166722340994121
chunk : 0.9211388421914739
conlleval : 92.04
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.024880031525361637 sec

```

- experiments 1-1
```
* test 10
word embedding size : 100 -> 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token) -> f1(chunk)
+
CRF

token : 0.9141382082857263
chunk : 0.9221816596741677
conlleval : 92.22

* test 9
word embedding size : 100
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF

token : 0.8917154244043076
chunk : 0.9032909170688899
conlleval : 90.33
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.028110911804991288 sec


* test 8
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 100 -> 50
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 25
pos embedding size : 65 -> 7
pos embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 512 -> 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF

token : 0.9133262846267709
chunk : 0.9209801552311596
conlleval : 92.09

* test 7
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 100
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 25
pos embedding size : 64 -> 65
pos embedding random init : -0.5 ~ 0.5
#pos one-hot : 5
#shape vec : 9
rnn_used : True
rnn_type : fused
rnn_size : 512
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF

token : 0.9137324987506671
chunk : 0.9220215504078872
conlleval : 92.20

* test 6
word embedding size : 300
keep_prob : 0.8 -> 0.7
chr_conv_type : conv1d
chracter embedding size : 100
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 50 -> 25
pos embedding size : 64
pos embedding random init : -0.5 ~ 0.5
#pos one-hot : 5
#shape vec : 9
rnn_used : True
rnn_type : fused
rnn_size : 512
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF

token : 0.9108174851701928
chunk : 0.9185380985669346
conlleval : 91.85

* test 5
word embedding size : 300
keep_prob : 0.8 -> 0.7
chr_conv_type : conv1d
chracter embedding size : 100
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 50
pos embedding size : 64
pos embedding random init : -0.5 ~ 0.5
#pos one-hot : 5
#shape vec : 9
rnn_used : True
rnn_type : fused
rnn_size : 512
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70 -> 100
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF

token : 0.8764648074588532
chunk : 0.8922909407665505
conlleval : 85.50

* test 4
word embedding size : 300
keep_prob : 0.8
chr_conv_type : conv1d
chracter embedding size : 100
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 50
pos embedding size : 64
pos embedding random init : -0.5 ~ 0.5
#pos one-hot : 5
#shape vec : 9
rnn_used : True
rnn_type : fused
rnn_size : 512
rnn_num_layers : 2 -> 3
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF

token : 0.8742414248021109
chunk : 0.8944653470745406
conlleval : 85.44

* test 3
word embedding size : 300
keep_prob : 0.8
chr_conv_type : conv1d
chracter embedding size : 100
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 50
pos embedding size : 64
pos embedding random init : -0.5 ~ 0.5
#pos one-hot : 5
#shape vec : 9
rnn_used : True
rnn_type : fused
rnn_size : 512
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20 -> 40
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF

token : 0.882063657166632
chunk : 0.8958401429474189
conlleval : 86.34

* test 2
word embedding size : 300
keep_prob : 0.8
chr_conv_type : conv1d
chracter embedding size : 100
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 50
pos embedding size : 64
pos embedding random init : -0.5 ~ 0.5
#pos one-hot : 5
#shape vec : 9
rnn_used : True
rnn_type : fused
rnn_size : 512
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF

token : 0.8821639702594843
chunk : 0.9003299032138253
conlleval : 86.30

* test 1
word embedding size : 300
keep_prob : 0.8
#chr_conv_type : conv1d
#chracter embedding size : 100
#chracter embedding random init : -1.0 ~ 1.0
#filter_sizes : [3]
#num_filters : 50
pos embedding size : 64
pos embedding random init : -0.5 ~ 0.5
#pos one-hot : 5
#shape vec : 9
rnn_used : True
rnn_type : fused
rnn_size : 512
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.7 -> 0.9
gradient clipping : 10
epoch : 70
batch_size : 20
+
tf_used : False
tf_keep_prob : 0.8
tf_mh_num_layers : 4
tf_mh_num_heads : 4
tf_mh_num_units : 64
tf_mh_keep_prob : 0.8
tf_ffn_keep_prob : 0.8
tf_ffn_kernel_size : 3
+
save model by f1(token)
+
CRF

token : 0.8786809180545199
chunk : 0.8954571291154737

```
