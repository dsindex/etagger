
- experiments 3
```
* test 3
word embedding size : 300
pos vec
pos embedding size : 5
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
shape vec
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.0001 -> 0.0002 (warmup), intermid_epoch = 15
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
pos_keep_prob : 0.5
epoch : 50
batch_size : 128
+
multi head attention(softmax with masking)
mh_num_heads : 2
mh_num_units : 32
mh_dropout : 0.2


* test 2
word embedding size : 300
pos vec
pos embedding size : 5
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
shape vec
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
pos_keep_prob : 0.5
epoch : 50
batch_size : 128
+
multi head attention
mh_num_heads : 2
mh_linear_key_dim : 32
mh_linear_val_dim : 32
mh_dropout : 0.5

0.896346749226 -> best

* test 1
word embedding size : 300
pos vec
pos embedding size : 5
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
shape vec
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
pos_keep_prob : 0.5
epoch : 50
batch_size : 128
+
multi head attention
mh_num_heads : 1
mh_linear_key_dim : 32
mh_linear_val_dim : 32
mh_dropout : 0.5

0.894368789106

```

- experiments 2
```
* test 17
word embedding size : 300
pos vec
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
shape vec
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001 -> 0.002 (warmup), intermid_epoch = 15
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 128

0.890206249228

* test 17
word embedding size : 300
pos vec
pos embedding size : 5
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
shape vec
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001 -> 0.002 (warmup), intermid_epoch = 15
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 128

0.893637926799

* test 16
word embedding size : 300
pos vec
pos embedding size : 5
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001 -> 0.002 (warmup), intermid_epoch = 15
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 128

0.891214677092

* test 15
word embedding size : 300
remove pos vec
pos embedding size : 10
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001 -> 0.002 (warmup), intermid_epoch = 15
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 128

0.887810272794

* test 14
word embedding size : 300
remove pos vec
pos embedding size : 50
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001 -> 0.002 (warmup), intermid_epoch = 15
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 128

0.886979997546

* test 13
word embedding size : 300
word embedding trainable : True
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001 -> 0.002 (warmup), intermid_epoch = 15
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 128

0.859092036715


* test 12
word embedding size : 300
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001 -> 0.002 (warmup), intermid_epoch = 15
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 128

0.895851286471

* test 11
word embedding size : 300
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 128
+
longest matching gazetteer feature(ignore length less than 10)

0.876311566473

* test 10
word embedding size : 300
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001 -> 0.002 (warmup), intermid_step = 1000
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 128
+
longest matching gazetteer feature(ignore length less than 10)

0.888393410133

* test 9
word embedding size : 300
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.0001 -> 0.001 (warmup), intermid_step = 1000
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 128

0.886325787948

* test 8
word embedding size : 300
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 128
+
longest matching gazetteer feature(without MISC)

0.866472158421

* test 7
word embedding size : 300
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 128

0.892918613228

* test 6
replace all digit to '0'
word embedding size : 300
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 128
+
longest matching gazetteer feature(from test data)

0.913716137712

=> ok. this result supports that gazetteer features are very helpful. but, 
if we construct gazetteer vocab from the training data, the f-score decreases.

* test 5
replace all digit to '0'
word embedding size : 300
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 128
+
longest matching gazetteer feature

0.870375031462

* test 4
replace all digit to '0'
word embedding size : 300
chracter embedding size : 96
chracter embedding random init : -1.0 ~ 1.0
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 128

0.890053001356

* test 3
replace all digit to '0'
random shuffling
word embedding size : 300
chracter embedding size : 53
chracter embedding random init : -1.0 ~ 1.0
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001
cnn_keep_prob : 0.5
rnn_keep_prob : 0.5
epoch : 50
batch_size : 64
+
longest matching gazetteer feature

0.87932312253

* test 2
replace all digit to '0'
random shuffling
word embedding size : 50
chracter embedding size : 64
chracter embedding random init : -0.5 ~ 0.5
filter_size : 3,4,5
num_filters : 32
rnn_size : 256
num_layers : 2
learning_rate : 0.001
cnn_keep_prob : 0.32
rnn_keep_prob : 0.32
epoch : 50
batch_size : 64

0.881000551775

* test 1
replace all digit to '0'
random shuffling
word embedding size : 50
chracter embedding size : 64
chracter embedding random init : -0.5 ~ 0.5
filter_size : 3
num_filters : 48
rnn_size : 256
num_layers : 1
learning_rate : 0.001
cnn_keep_prob : 0.32
rnn_keep_prob : 0.32
epoch : 64
batch_size : 64

0.884797152151

```

- experiments 1
```
* weak entity types : B-ORG, I-ORG, B-MISC, I-MISC

* chr_embedding : max

rnn_size : 256, keep_prob : 0.5, chr_embedding : max
0.892409321671

* chr embedding : conv

rnn_size : 256, keep_prob : 0.5, chr_embedding : conv
0.895172667607
0.893800406329
0.892967114177
0.893781430148

rnn_size : 256, cnn_keep_prob : 0.7, rnn_keep_prob : 0.8, chr_embedding : conv
0.892371739929

rnn_size : 256, cnn_keep_prob : 0.6, rnn_keep_prob : 0.6, chr_embedding : conv
0.893224198412

* gazetteer feature

rnn_size : 256, keep_prob : 0.5, chr_embedding : conv, gazetteer : token-based m-hot vector
0.855807086614

rnn_size : 512, keep_prob : 0.5, chr_embedding : conv, gazetteer : token-based m-hot vector
0.873537604457

rnn_size : 256, keep_prob : 0.5, chr_embedding : conv, gazetteer : token-based 0|1
0.877048661647

even though we use '0|1' indicating gazetteer, it is worse than basic models.
the loss is even increasing along steps. why?

try to adjust keep_probs.
rnn_size : 256, cnn_keep_prob : 0.8, rnn_keep_prob : 0.8, chr_embedding : conv, gazetteer : token-based 0|1
0.879918632001

try to filter digit/ascii symbol/short word from gazetteer vocab.
rnn_size : 256, cnn_keep_prob : 0.8, rnn_keep_prob : 0.8, chr_embedding : conv, gazetteer : token-based 0|1
0.877144298688

use m-hot vector and apply unambiguous gazetteer only
rnn_size : 256, cnn_keep_prob : 0.8, rnn_keep_prob : 0.8, chr_embedding : conv, gazetteer : token-based m-hot vector
0.883349826818

including unambiguous 'O' gazetteer
rnn_size : 256, cnn_keep_prob : 0.8, rnn_keep_prob : 0.8, chr_embedding : conv, gazetteer : token-based m-hot vector
0.878849345381
```
