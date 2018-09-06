etagger
====

### description

- named entity tagger using multi-layer Bidirectional LSTM

- original git repository
  - https://github.com/monikkinom/ner-lstm

- modification
  - modified for tf version(1.4)
  - removed unnecessary files
  - fixed bugs for MultiRNNCell()
  - refactoring .... ing
    - implement input.py, config.py [done]
    - split model.py to model.py, train.py, inference.py [done]
      - inference bulk [done]
      - inference bucket [done]
      - inference line using spacy [done]
    - extend 5 tag(class) to 9 (automatically) [done]
      - out of tag(class) 'O' and its id '0' are fixed
    - apply dropout for train() only [done]
    - apply embedding_lookup()
      - word embedding [done]
    - apply character-level embedding
      - using convolution [done]
      - using reduce_max only [done]
      ![graph-1](https://raw.githubusercontent.com/dsindex/etagger/master/etc/graph-1.png)
    - apply gazetter features [doing]
    - apply self-attention
    - apply ELMO embedding
    - serve api
- references
  - https://web.stanford.edu/class/cs224n/reports/6896582.pdf
  - http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
  - https://github.com/cuteboydot/Sentence-Classification-using-Char-CNN-and-RNN
  - https://github.com/carpedm20/lstm-char-cnn-tensorflow/blob/master/models/LSTMTDNN.py
  - https://github.com/DongjunLee/transformer-tensorflow/blob/master/transformer/attention.py

### pre-requisites

- data
  - [download](https://github.com/mxhofer/Named-Entity-Recognition-BidirectionalLSTM-CNN-CoNLL/tree/master/data) 
  - place train.txt, dev.txt, test.txt in data dir

- glove embedding
  - [download](http://nlp.stanford.edu/data/glove.6B.zip)
  - unzip to 'embeddings' dir

- spacy [optional]
  - if you want to analyze input string and see how it detects entities, then you need to install spacy lib.
  ```
  $ pip install spacy
  $ python -m spacy download en
  ```

### how to 

- convert word embedding to pickle
```
$ python embvec.py --emb_path embeddings/glove.6B.50d.txt --wrd_dim 50 --train_path data/train.txt
$ python embvec.py --emb_path embeddings/glove.6B.100d.txt --wrd_dim 100 --train_path data/train.txt
$ python embvec.py --emb_path embeddings/glove.6B.300d.txt --wrd_dim 300 --train_path data/train.txt
```

- check max sentence length
```
$ python check_sentence_length.py
train, max_sentence_length = 113
dev, max_sentence_length = 109
test, max_sentence_length = 124

* set 125 to sentence_length
```

- train
```
$ python train.py --emb_path embeddings/glove.6B.300d.txt.pkl --wrd_dim 300 --sentence_length 125
$ rm -rf runs; tensorboard --logdir runs/summaries/
```

- inference(bulk)
```
$ python inference.py --emb_path embeddings/glove.6B.300d.txt.pkl --wrd_dim 300 --sentence_length 125 --restore checkpoint/model_max.ckpt
```

- inference(bucket)
```
$ python inference.py --mode bucket --emb_path embeddings/glove.6B.300d.txt.pkl --wrd_dim 300 --sentence_length 125 --restore checkpoint/model_max.ckpt < data/test.txt > pred.txt
$ python eval.py < pred.txt
```

- inference(line)
```
$ python inference.py --mode line --emb_path embeddings/glove.6B.300d.txt.pkl --wrd_dim 300 --sentence_length 125 --restore checkpoint/model_max.ckpt
...
Obama left office in January 2017 with a 60% approval rating and currently resides in Washington, D.C.
Obama NNP O O B-PER
left VBD O O O
office NN O O O
in IN O O O
January NNP O B-DATE O
2017 CD O I-DATE O
with IN O O O
a DT O O O
60 CD O B-PERCENT O
% NN O I-PERCENT O
approval NN O O O
rating NN O O O
and CC O O O
currently RB O O O
resides VBZ O O O
in IN O O O
Washington NNP O B-GPE B-LOC
, , O I-GPE O
D.C. NNP O B-GPE I-LOC

The Beatles were an English rock band formed in Liverpool in 1960.
The DT O O O
Beatles NNPS O B-PERSON O
were VBD O O O
an DT O O O
English JJ O B-LANGUAGE B-MISC
rock NN O O O
band NN O O O
formed VBN O O O
in IN O O O
Liverpool NNP O B-GPE B-ORG
in IN O O O
1960 CD O B-DATE O
. . O I-DATE O
```

### etc

- analysis
```
* weak entity types : B-ORG, I-ORG, B-MISC, I-MISC

* chr_embedding : max
rnn_size : 256, dropout_rate : 0.5, chr_embedding : max
0.892409321671

* chr embedding : conv
rnn_size : 256, dropout_rate : 0.5, chr_embedding : conv
0.895172667607 <-- best
0.893800406329
0.892967114177
rnn_size : 512, dropout_rate : 0.15, chr_embedding : conv
0.878538026089
rnn_size : 128, dropout_rate : 0.3, chr_embedding : conv
0.886479827533

* gazetteer feature
rnn_size : 256, dropout_rate : 0.5, chr_embedding : conv, gazetteer : 1/0s vector
0.855807086614
rnn_size : 512, dropout_rate : 0.5, chr_embedding : conv, gazetteer : 1/0s vector
0.873537604457
rnn_size : 512, dropout_rate : 0.3, chr_embedding : conv, gazetteer : count vector
0.849502098077
rnn_size : 128, dropout_rate : 0.3, chr_embedding : conv, gazetteer : 1/0s vector
0.856015779093

even if we use '0|1' indicating gazetteer or not, it is worse than basic models. why?
the loss is even increasing along steps.
rnn_size : 256, dropout_rate : 0.5, chr_embedding : conv, gazetteer : 0|1
0.877048661647
```
