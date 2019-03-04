- summary
  - https://docs.google.com/spreadsheets/d/1Sy7YREtqsIaaesNM1LAdsstomB7nrQV483XPzwH5JBM/edit?usp=sharing

- experiments 2 data
```
1. number of labels
485

2. data
25M, 109120 sentences, data/cruise.train.txt
3.1M, 13640 sentences, data/cruise.dev.txt
3.1M, 13642 sentences, data/cruise.test.txt

* estimated pre-processing time
morpological analysis : 7 sec / 137272 =  0.000050993647648 sec
dependency parsing    : 20.954375 sec / 137272 = 0.000152648573635 sec
base entity tagging   : 230.612970 sec / 137272 = 0.001679970933621 sec

3. glove
2.5G(500k, 525470) kor.glove.300d.txt
841M(300k, 308383) kor.glove.300k.300d.txt

4. bert

1) all.dha.2.5m_step
  768    hidden_size
  202592 vocab_size
  2.5G   bert_model.ckpt
  64     bert_max_seq_length

2) multi_cased_L-12_H-768_A-12
  768    hidden_size
  119547 vocab_size
  682M   bert_model.ckpt
  96     bert_max_seq_length

3) all.200k.out.1m-step.reduced
  768    hidden_size
  100795 vocab_size
  627M   bert_model.ckpt
  96     bert_max_seq_length

5. evaluation by CRF(wapiti)
token : 0.852890598904
chunk : 0.844539652006645
conlleval : 84.45
average processing time per bucket(sentence) : 227.275150 / 13642 = 0.01665995821727 sec
```

- how to run
```
$ cd data
$ python ../etc/conv.py < cruise.train.txt > cruise.train.txt.in
$ python ../etc/conv.py < cruise.dev.txt > cruise.dev.txt.in
$ python ../etc/conv.py < cruise.test.txt > cruise.test.txt.in
$ cat cruise.train.txt.in cruise.dev.txt.in cruise.test.txt.in > cruise.total.txt.in

- embedding
* Glove
$ python embvec.py --emb_path embeddings/kor.glove.100d.txt --wrd_dim 100 --train_path data/cruise.train.txt.in --total_path data/cruise.total.txt.in > embeddings/vocab.txt
$ python embvec.py --emb_path embeddings/kor.glove.300d.txt --wrd_dim 300 --train_path data/cruise.train.txt.in --total_path data/cruise.total.txt.in > embeddings/vocab.txt
$ python embvec.py --emb_path embeddings/kor.glove.300k.300d.txt --wrd_dim 300 --train_path data/cruise.train.txt.in --total_path data/cruise.total.txt.in > embeddings/vocab.txt

* for BERT(all.dha.2.5m_step)
$ python embvec.py --emb_path embeddings/kor.glove.300d.txt --wrd_dim 300 --train_path data/cruise.train.txt.in --total_path data/cruise.total.txt.in --bert_config_path embeddings/all.dha.2.5m_step/bert_config.json --bert_vocab_path embeddings/all.dha.2.5m_step/vocab.txt --bert_do_lower_case False --bert_init_checkpoint embeddings/all.dha.2.5m_step/bert_model.ckpt --bert_max_seq_length 64 > embeddings/vocab.txt
$ python embvec.py --emb_path embeddings/kor.glove.300k.300d.txt --wrd_dim 300 --train_path data/cruise.train.txt.in --total_path data/cruise.total.txt.in --bert_config_path embeddings/all.dha.2.5m_step/bert_config.json --bert_vocab_path embeddings/all.dha.2.5m_step/vocab.txt --bert_do_lower_case False --bert_init_checkpoint embeddings/all.dha.2.5m_step/bert_model.ckpt --bert_max_seq_length 64 > embeddings/vocab.txt

* for BERT(multi_cased_L-12_H-768_A-12)
$ python embvec.py --emb_path embeddings/kor.glove.300k.300d.txt --wrd_dim 300 --train_path data/cruise.train.txt.in --total_path data/cruise.total.txt.in --bert_config_path embeddings/multi_cased_L-12_H-768_A-12/bert_config.json --bert_vocab_path embeddings/multi_cased_L-12_H-768_A-12/vocab.txt --bert_do_lower_case False --bert_init_checkpoint embeddings/multi_cased_L-12_H-768_A-12/bert_model.ckpt --bert_max_seq_length 96 > embeddings/vocab.txt

* for BERT(all.200k.out.1m-step.reduced)
$ python embvec.py --emb_path embeddings/kor.glove.300k.300d.txt --wrd_dim 300 --train_path data/cruise.train.txt.in --total_path data/cruise.total.txt.in --bert_config_path embeddings/all.200k.out.1m-step.reduced/bert_config.json --bert_vocab_path embeddings/all.200k.out.1m-step.reduced/vocab.txt --bert_do_lower_case False --bert_init_checkpoint embeddings/all.200k.out.1m-step.reduced/bert_model.ckpt --bert_max_seq_length 96 > embeddings/vocab.txt

- train
$ python train.py --emb_path embeddings/kor.glove.100d.txt.pkl --wrd_dim 100 --batch_size 20 --epoch 70
$ python train.py --emb_path embeddings/kor.glove.300d.txt.pkl --wrd_dim 300 --batch_size 20 --epoch 70
$ python train.py --emb_path embeddings/kor.glove.300k.300d.txt.pkl --wrd_dim 300 --batch_size 20 --epoch 70

- inference
$ python inference.py --mode bucket --emb_path embeddings/kor.glove.100d.txt.pkl --wrd_dim 100 --restore checkpoint/ner_model < data/cruise.test.txt.in > pred.txt
$ python inference.py --mode bucket --emb_path embeddings/kor.glove.300d.txt.pkl --wrd_dim 300 --restore checkpoint/ner_model < data/cruise.test.txt.in > pred.txt
$ python inference.py --mode bucket --emb_path embeddings/kor.glove.300k.300d.txt.pkl --wrd_dim 300 --restore checkpoint/ner_model < data/cruise.test.txt.in > pred.txt
```

- experiments 2-2
```

* test 5
word embedding size : 300 -> 300(kor.glove.300k.300d.txt)
#bert embedding : all.200k.out.1m-step.reduced
#bert_keep_prob : 0.8
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
highway_used : False -> True
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
#learning_rate : use optimization.py from bert, 2e-5 / warmup proportion 0.1
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

token : 0.8629193131891667
chunk : 0.8560444070066309
conlleval : 85.60          -> Glove + CNN + CHK + HIGHWAY + LSTM + CRF best
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.013156061647155387 sec
  - 32 CPU : skip
  - 1 CPU : skip

* test 4
word embedding size : 300 -> 300(kor.glove.300k.300d.txt)
bert embedding : all.200k.out.1m-step.reduced
bert_keep_prob : 0.8
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200 -> 256
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
#learning_rate : use optimization.py from bert, 2e-5 / warmup proportion 0.1
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

token : 0.8634652061225488
chunk : 0.856661773594681
conlleval : 85.66          -> Glove + BERT + CNN + CHK + LSTM + CRF best
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.024302998467845102 sec
  - 56 CPU : skip
  - 1 CPU : skip

* test 3
word embedding size : 300 -> 300(kor.glove.300k.300d.txt)
bert embedding : multi_cased_L-12_H-768_A-12
bert_keep_prob : 0.8
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200 -> 256
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
#learning_rate : use optimization.py from bert, 2e-5 / warmup proportion 0.1
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

token : 0.861540358882654
chunk : 0.8465366977461912
conlleval : 83.89
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.027959363024669247 sec
  - 56 CPU : skip
  - 1 CPU : skip

* test 2
word embedding size : 300 -> 300(kor.glove.300k.300d.txt)
bert embedding : all.dha.2.5m_step
bert_keep_prob : 0.8
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200 -> 256
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
#learning_rate : use optimization.py from bert, 2e-5 / warmup proportion 0.1
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

token : 0.8618997374426423
chunk : 0.8556839326669262
conlleval : 85.56
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.021882983845116683 sec
  - 56 CPU : skip
  - 1 CPU : skip

* test 1
word embedding size : 300 -> 300(kor.glove.300k.300d.txt)
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
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

token : 0.863193851409052
chunk : 0.8553803679658347
conlleval : 85.54          -> Glove + CNN + CHK + LSTM + CRF best
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.013301651632658217 sec
  - 32 CPU : skip
  - 1 CPU : skip

```

- experiments 2-1
```

* test 15
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2 -> 1
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9 -> 0.001 / 3000 / 0.9
gradient clipping : 10 -> 1.5
epoch : 70 -> 140
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
save model by f1(token) -> f1(chunk)
+
CRF

token : 0.8613488783943329
chunk : 0.8521689993914632
conlleval : 85.21
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.010911516509890085 sec
  - 32 CPU : skip
  - 1 CPU : skip

* test 14
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9 -> 0.001 / 3000 / 0.9
gradient clipping : 10 -> 1.5
epoch : 70 -> 140
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
save model by f1(token) -> f1(chunk)
+
CRF

token : 0.8612126714597698
chunk : 0.8534114329102941
conlleval : 85.34
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.009480409023786567 sec
  - 56 CPU : skip
  - 1 CPU : skip

* test 13
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9 -> 0.001 / 5000 / 0.9
gradient clipping : 10 -> 1.5
epoch : 70 -> 140
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

token : 0.8609414477395934
chunk : 0.8521701482844464
conlleval : 85.21
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.009061403191346538 sec
  - 56 CPU : skip
  - 1 CPU : skip

* test 12
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9 -> 0.001 / 3000 / 0.9
gradient clipping : 10 -> 1.5
epoch : 70 -> 140
batch_size : 20 -> 100
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
#CRF

token : 0.859288847501654
chunk : 0.8433297297297296
conlleval : 83.67
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.005661899070728561 sec
  - 32 CPU : skip
  - 1 CPU : skip

* test 11
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9 -> 0.001 / 3000 / 0.9
gradient clipping : 10 -> 1.5
epoch : 70 -> 140
batch_size : 20 -> 100
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

token : 0.8603671970624236
chunk : 0.8519950744237292
conlleval : 85.20
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.011507198006993881 sec
  - 56 CPU : skip
  - 1 CPU : skip

* test 10
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9 -> 0.001 / 3000 / 0.9
gradient clipping : 10 -> 1.5
epoch : 70 -> 140
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
save model by f1(token) -> f1(chunk)
+
#CRF

token : 0.859759461546949
chunk : 0.8433318200488574
conlleval : 83.72
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.005789112020597974 sec
  - 32 CPU : skip
  - 1 CPU : skip

* test 9
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9 -> 0.0003 / 3000 / 0.9
gradient clipping : 10 -> 1.5
epoch : 70 -> 140
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
save model by f1(token) -> f1(chunk)
+
CRF

token : 0.8584552485099256
chunk : 0.8484334203655352
conlleval : 84.84
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.01205325771542068 sec
  - 56 CPU : skip
  - 1 CPU : skip

* test 8
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9 -> 0.001 / 3000 / 0.9
gradient clipping : 10 -> 1.5
epoch : 70 -> 140
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
save model by f1(token) -> f1(chunk)
+
CRF

# test data
token : 0.8632774180890719
chunk : 0.8540854345377351
conlleval :  85.40
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) :  0.013347263088710076 sec
  - 32 CPU : skip
  - 1 CPU : skip

# dev data
token : 0.8616541715883616
chunk : 0.8557254183877699
conlleval : 85.57

* test 7
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9 -> 2e-5 / 30000 / 0.9
gradient clipping : 10 -> 1.5
epoch : 70 -> 140
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
save model by f1(token) -> f1(chunk)
+
CRF

token : 0.8520733455246379
chunk : 0.8389521291703992
conlleval : 83.89
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.011398896009162485 sec
  - 56 CPU : skip
  - 1 CPU : skip

* test 6
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
#learning_rate : exponential_decay(), 0.001 / 12000 / 0.9
#gradient clipping : 10 -> 1.5
use bert optimization : 2e-5, warmup proportion 0.05
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

token : 0.848494606343584
chunk : 0.8331763314127982
conlleval : 83.32
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.013032469279674155 sec
  - 32 CPU : skip
  - 1 CPU : skip

* test 5
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9 -> 2e-5 / 50000 / 0.9
gradient clipping : 10 -> 1.5
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

token : 0.8528737332642947
chunk : 0.8399615837953464
conlleval : 83.99
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.011336098369519534 sec
  - 56 CPU : skip
  - 1 CPU : skip

* test 4
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64 -> 128
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200 -> 256
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9 -> 0.0003 / 10000 / 0.9
gradient clipping : 10
epoch : 70
batch_size : 20 -> 10
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

token : 0.8616240020665986
chunk : 0.852711157455683
conlleval : 85.26
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.010170429774593212 sec
  - 56 CPU : skip
  - 1 CPU : skip

* test 3
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9 -> 0.0003 / 10000 / 0.9
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

token : 0.8618139449316244, 0.9153326218427605(including 'O')
chunk : 0.8528806807110225, 0.9259648466178184(including 'O')
conlleval : 85.28         , 92.60(including 'O')
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.011372548012313317 sec
  - 56 CPU : 0.010189664309682987 sec
  - 1 CPU : 0.011781872515978307 sec

* test 2
word embedding size : 300
keep_prob : 0.7 -> 0.5
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
rnn_used : True
rnn_type : fused
rnn_size : 200 -> 256
rnn_num_layers : 2
learning_rate : exponential_decay(), 0.001 / 12000 / 0.9 -> 2e-5 / 5000 / 0.9
gradient clipping : 10 -> 1.5
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
save model by f1(token) -> f1(chunk)
+
CRF

token : 0.8230523531466787
chunk : 0.7995750237942408
conlleval : 79.96
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.010178041622411853 sec
  - 56 CPU : skip
  - 1 CPU : skip

* test 1
word embedding size : 300
keep_prob : 0.7
chr_conv_type : conv1d
chracter embedding size : 25
chracter embedding random init : -1.0 ~ 1.0
filter_sizes : [3]
num_filters : 53
pos embedding size : 7
pos embedding random init : -0.5 ~ 0.5
chk embedding size : 10 -> 64
chk embedding random init : -0.5 ~ 0.5
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

token : 0.8611652747630085
chunk : 0.8521355316110607
conlleval : 85.21
average processing time per bucket(sentence)
  - 1 GPU(V100 TESLA) : 0.011398719440997651 sec
  - 56 CPU : 0.010089373766825427 sec
  - 1 CPU : 0.011839326342745028 sec
```


- experiments 1 data
```
1. number of labels
485

2. data
15M  data/cruise.train.txt.in
1.8M ata/cruise.dev.txt.in
1.8M data/cruise.test.txt.in

3. glove
2.5G(500k, 525470)  kor.glove.300d.txt

4. evaluation by CRF(wapiti)
token : 0.820348045768
chunk : 0.8155083158938209
conlleval : 81.46
average processing time per bucket(sentence) : 235.691333 / 13692 = 0.017213798787613 sec
```

- how to run
```
$ python embvec.py --emb_path embeddings/kor.glove.100d.txt --wrd_dim 100 --train_path data/cruise.train.txt.in --total_path data/cruise.total.txt.in > embeddings/vocab.txt
$ python embvec.py --emb_path embeddings/kor.glove.300d.txt --wrd_dim 300 --train_path data/cruise.train.txt.in --total_path data/cruise.total.txt.in > embeddings/vocab.txt

$ python train.py --emb_path embeddings/kor.glove.100d.txt.pkl --wrd_dim 100 --batch_size 20 --epoch 70
$ python train.py --emb_path embeddings/kor.glove.300d.txt.pkl --wrd_dim 300 --batch_size 20 --epoch 70

$ python inference.py --mode bucket --emb_path embeddings/kor.glove.100d.txt.pkl --wrd_dim 100 --restore checkpoint/ner_model < data/cruise.test.txt.in > pred.txt
$ python inference.py --mode bucket --emb_path embeddings/kor.glove.300d.txt.pkl --wrd_dim 300 --restore checkpoint/ner_model < data/cruise.test.txt.in > pred.txt
```

- experiments 1-1
```
* test 1
word embedding size : 300
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

token : 0.8336974003216593, 0.8950232838811782(including 'O')
chunk : 0.828596112311015 , 0.9093912290825158(including 'O')
conlleval : 82.78         , 90.93(including 'O')              -> Glove + CNN + LSTM + CRF best
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.01303002712777439 sec
  - 32 CPU : 0.011279379889660678 sec
  - 1 CPU : 0.010049216377432151 sec
```

