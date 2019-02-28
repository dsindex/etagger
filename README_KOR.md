
- experiments 3 data
```
1. number of labels
485

2. data
25M, 109120 sentences, data/cruise.train.txt
3.1M, 13640 sentences, data/cruise.dev.txt
3.1M, 13642 sentences, data/cruise.test.txt

3. glove
2.5G   kor.glove.300d.txt
525470 embeddings/vocab.txt

4. evaluation by CRF(wapiti)
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

$ python embvec.py --emb_path embeddings/kor.glove.100d.txt --wrd_dim 100 --train_path data/cruise.train.txt.in --total_path data/cruise.total.txt.in > embeddings/vocab.txt
$ python embvec.py --emb_path embeddings/kor.glove.300d.txt --wrd_dim 300 --train_path data/cruise.train.txt.in --total_path data/cruise.total.txt.in > embeddings/vocab.txt

$ python train.py --emb_path embeddings/kor.glove.100d.txt.pkl --wrd_dim 100 --batch_size 40 --epoch 140
$ python train.py --emb_path embeddings/kor.glove.300d.txt.pkl --wrd_dim 300 --batch_size 40 --epoch 140

$ python inference.py --mode bucket --emb_path embeddings/kor.glove.100d.txt.pkl --wrd_dim 100 --restore checkpoint/ner_model < data/cruise.test.txt.in > pred.txt
$ python inference.py --mode bucket --emb_path embeddings/kor.glove.300d.txt.pkl --wrd_dim 300 --restore checkpoint/ner_model < data/cruise.test.txt.in > pred.txt
```

- experiments 3-1
```

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
  - 8 CPU : skip
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
  - 8 CPU : skip
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
conlleval :  85.40 -> best
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
  - 8 CPU : skip
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
  - 8 CPU : skip
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
  - 8 CPU : skip
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
  - 8 CPU : 0.010189664309682987 sec
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
  - 8 CPU : skip
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
  - 8 CPU : 0.010089373766825427 sec
  - 1 CPU : 0.011839326342745028 sec
```


- experiments 2 data
```
1. number of labels
485

2. data
15M  data/cruise.train.txt.in
1.8M ata/cruise.dev.txt.in
1.8M data/cruise.test.txt.in

3. glove
2.5G   kor.glove.300d.txt
525470 embeddings/vocab.txt


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

- experiments 2-1
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
conlleval : 82.78         , 90.93(including 'O') -> best
average processing time per bucket(sentence)
  - 1 GPU(TITAN X PASCAL) : 0.01303002712777439 sec
  - 32 CPU : 0.011279379889660678 sec
  - 1 CPU : 0.010049216377432151 sec
```

- experiments 1 data
```
1. number of labels
610

2. data
10M  data/kor.train.txt
1.7M ata/kor.dev.txt
2.4M data/kor.test.txt

3. glove
2.5G   kor.glove.300d.txt
525470 embeddings/vocab.txt

4. evaluation by CRF(wapiti)
token : 0.900468066343
chunk : 0.9141800490902886 
conlleval : 91.42

```

- how to run
```
$ python embvec.py --emb_path embeddings/kor.glove.100d.txt --wrd_dim 100 --train_path data/kor.train.txt --total_path data/kor.total.txt > embeddings/vocab.txt
$ python embvec.py --emb_path embeddings/kor.glove.300d.txt --wrd_dim 300 --train_path data/kor.train.txt --total_path data/kor.total.txt > embeddings/vocab.txt

$ python train.py --emb_path embeddings/kor.glove.100d.txt.pkl --wrd_dim 100 --batch_size 20 --epoch 70
$ python train.py --emb_path embeddings/kor.glove.300d.txt.pkl --wrd_dim 300 --batch_size 20 --epoch 70

$ python inference.py --mode bucket --emb_path embeddings/kor.glove.100d.txt.pkl --wrd_dim 100 --restore checkpoint/ner_model < data/kor.test.txt > pred.txt
$ python inference.py --mode bucket --emb_path embeddings/kor.glove.300d.txt.pkl --wrd_dim 300 --restore checkpoint/ner_model < data/kor.test.txt > pred.txt
```

- experimements 1-1
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
conlleval : 92.22          -> best

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
