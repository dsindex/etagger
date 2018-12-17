- experiments data
```
1. number of labels
610

2. data
-rw-r--r-- 1 hanadmin hanmail  10M 12월 11 16:25 data/kor.train.txt
-rw-r--r-- 1 hanadmin hanmail 2.4M 12월 11 16:25 data/kor.test.txt
-rw-r--r-- 1 hanadmin hanmail 1.7M 12월 11 16:25 data/kor.dev.txt

3. glove
-rw-r--r--  1 hanadmin hanmail 1.2G 12월 14 10:52 kor.glove.300d.txt.pkl
-rw-r--r--  1 hanadmin hanmail 2.5G 12월 14 10:49 kor.glove.300d.txt

525470 embeddings/vocab.txt
```

- how to run
```
$ python embvec.py --emb_path embeddings/kor.glove.100d.txt --wrd_dim 100 --train_path data/kor.train.txt --total_path data/kor.total.txt --elmo_vocab_path embeddings/elmo_vocab.txt --elmo_options_path embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json --elmo_weight_path embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 > embeddings/vocab.txt
$ python embvec.py --emb_path embeddings/kor.glove.300d.txt --wrd_dim 300 --train_path data/kor.train.txt --total_path data/kor.total.txt --elmo_vocab_path embeddings/elmo_vocab.txt --elmo_options_path embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json --elmo_weight_path embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 > embeddings/vocab.txt


$ python train.py --emb_path embeddings/kor.glove.100d.txt.pkl --wrd_dim 100 --batch_size 20 --epoch 70
$ python train.py --emb_path embeddings/kor.glove.300d.txt.pkl --wrd_dim 300 --batch_size 20 --epoch 70

$ python inference.py --mode bucket --emb_path embeddings/kor.glove.100d.txt.pkl --wrd_dim 100 --restore checkpoint/ner_model < data/kor.test.txt > pred.txt
$ python inference.py --mode bucket --emb_path embeddings/kor.glove.300d.txt.pkl --wrd_dim 300 --restore checkpoint/ner_model < data/kor.test.txt > pred.txt
```

- experimements
```
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

token-based : 0.8786809180545199
chunk-based : 0.8954571291154737

```
