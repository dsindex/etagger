etagger
====

### description

- named entity tagger using
  - embedding
    - word embedding(glove) and character convolutional embedding
    - ELMO embedding
  - pos embedding
  - gazetteer features
  - multi-layer Bidirectional LSTM
  - multi-head attention
  - CRF decoder

- base repository
  - [ner-lstm](https://github.com/monikkinom/ner-lstm)
  - [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py)
  - [transformer/modules.py](https://github.com/Kyubyong/transformer/blob/master/modules.py)
  - [sequence_tagging/ner_model.py](https://github.com/guillaumegenthial/sequence_tagging/blob/master/model/ner_model.py)
  - [tf_ner/masked_conv.py](https://github.com/guillaumegenthial/tf_ner/blob/master/models/chars_conv_lstm_crf/masked_conv.py)
  - [bilm](https://github.com/allenai/bilm-tf)

- modification
  - refactoring
    - [x] config.py, input.py
    - [x] inference.py
      - inference bulk
      - inference bucket
      - inference line using spacy
    - [x] extend 5 tag(class) to 9 (automatically)
      - out of tag(class) 'O' and its id '0' are fixed
    - [x] word embedding(glove)
      - Glove6B(lowercase), Glove840B(no lowercase)
    - [x] character embedding
      - conv2d, conv1d(masked)
    - [x] gazetteer features
      - if we construct the gazetteer vocab from the training data, the performance is decreasing.
      - recommend constructing the gazetteer vocab from other sources.
    - [x] learning-rate decay
      - using tf.train.exponential_decay()
    - [x] pos embedding
      - pos one-hot vector + pos embedding
    - [x] extend language specific features
      - initialCaps, allCaps, lowercase, mixedCaps, no-info, ...
    - [x] bidirectional lstm
      - tf.nn.bidirectional_dynamic_rnn()
      - tf.contrib.rnn.LSTMBlockFusedCell()
    - [x] multi-head self-attention
      - softmax with query, key masking
    - [x] CRF
      - train by crf_log_likelihood()
      - inference by viterbi_decode()
    - [x] early stopping
    - [ ] IOBES tagging schemes
    - [ ] ELMO embedding
    - [ ] curriculum learning
      - sort the training data ascending order by average entropy(calculated at the end of layers) 
    - [ ] serve api
      - freeze model and serve

- model
  ![graph-1](https://raw.githubusercontent.com/dsindex/etagger/master/etc/graph-1.png)

- evaluation
  - [experiments](https://github.com/dsindex/etagger/blob/master/README_DEV.md)
  - fscore
    - experiments 6, test 2
      - 70 epoch, per-token(partial) micro f1 : 0.9138489758483644
      - 70 epoch, per-chunk(exact)   micro f1 : **0.9082082965578112**
  - comparision to previous research
    - implementations
      - [Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs](https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs)
        - tested
        - Glove6B.100
        - 70 epoch, per-chunk(exact) micro Prec: 0.887, Rec: 0.902, F1: 0.894
      - [sequence_tagging](https://github.com/guillaumegenthial/sequence_tagging)
        - tested
        - Glove6B.100
        - 15 epoch, per-chunk(exact) miscro F1: 0.8998
          - early stopping 3 epochs without improvement
      - [tf_ner](https://github.com/guillaumegenthial/tf_ner)
        - tested
        - Glove840B.300
        - F1 : 0.905 ~ 0.907 (chars_conv_lstm_crf)
          - reported F1 : 0.9121
    - SOTA
      - [Semi-Supervised Sequence Modeling with Cross-View Training](https://arxiv.org/pdf/1809.08370.pdf)
        - reported F1 : 0.926
      - [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)
        - reported F1 : 0.9222
      - [Semi-supervised sequence tagging with bidirectional language models](https://arxiv.org/pdf/1705.00108.pdf)
        - reported F1 : 0.9193

### pre-requisites

- python >= 3.6

- tensorflow >= 1.10
```
tensorflow 1.10, CUDA 9.0, cuDNN 7.12
(cuda-9.0-pkg/cuda-9.0/lib64, cudnn-9.0/lib64)

tensorflow 1.11, CUDA 9.0, cuDNN 7.31
(cuda-9.0-pkg/cuda-9.0/lib64, cudnn-9.0-v73/lib64)
```

- numpy

- data
  - [download data of CoNLL 2003 shared task](https://github.com/mxhofer/Named-Entity-Recognition-BidirectionalLSTM-CNN-CoNLL/tree/master/data) 
  - place train.txt, dev.txt, test.txt in data dir
  - merge to total.txt in data dir

- glove embedding
  - [download Glove6B](http://nlp.stanford.edu/data/glove.6B.zip)
  - [download Glove840B](http://nlp.stanford.edu/data/glove.840B.300d.zip)
  - unzip to 'embeddings' dir

- bilm
  - install [bilm](https://github.com/allenai/bilm-tf)
  - download [ELMO weights and options](https://allennlp.org/elmo)
  ```
  $ ls embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_*
  embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json  embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
  ``` 

- spacy [optional]
  - if you want to analyze input string and see how it detects entities, then you need to install spacy lib.
  ```
  $ pip install spacy
  $ python -m spacy download en
  ```

### how to 

- convert word embedding to pickle
```
$ python embvec.py --emb_path embeddings/glove.6B.50d.txt --wrd_dim 50 --train_path data/train.txt --total_path data/total.txt
$ python embvec.py --emb_path embeddings/glove.6B.100d.txt --wrd_dim 100 --train_path data/train.txt --total_path data/total.txt
$ python embvec.py --emb_path embeddings/glove.6B.200d.txt --wrd_dim 200 --train_path data/train.txt --total_path data/total.txt
$ python embvec.py --emb_path embeddings/glove.6B.300d.txt --wrd_dim 300 --train_path data/train.txt --total_path data/total.txt
$ python embvec.py --emb_path embeddings/glove.840B.300d.txt --wrd_dim 300 --train_path data/train.txt --total_path data/total.txt --lowercase 0
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
  - command
  ```
  $ python train.py --emb_path embeddings/glove.6B.100d.txt.pkl --wrd_dim 100 --sentence_length 125 --batch_size 20 --epoch 70
  $ rm -rf runs; tensorboard --logdir runs/summaries/ --port 6007
  ```
  - accuracy and loss
  ![graph-2](https://raw.githubusercontent.com/dsindex/etagger/master/etc/graph-2.png)
  - abnormal case when using multi-head
  ![graph-3](https://raw.githubusercontent.com/dsindex/etagger/master/etc/graph-3.png)
  - why? 
  ```
  i guess that the softmax(applied in multi-head attention functions) was corrupted by paddings.
    -> so, i replaced the multi-head attention code to `https://github.com/Kyubyong/transformer/blob/master/modules.py`
       which applies key and query masking for paddings.
    -> however, simillar corruption was happended.
    -> it was caused by the tf.contrib.layers.layer_norm() which normalizes over [begin_norm_axis ~ R-1] dimensions.
    -> what about remove the layer_norm()? performance goes down!
    -> try to use other layer normalization code from `https://github.com/Kyubyong/transformer/blob/master/modules.py`
       which normalizes over the last dimension only.
       this code perfectly matches to my intention.
  ```
  - after replacing layer_norm() to normalize() and applying the dropout of word embeddings
  ![graph-4](https://raw.githubusercontent.com/dsindex/etagger/master/etc/graph-4.png)

- inference(bulk)
```
$ python inference.py --emb_path embeddings/glove.6B.100d.txt.pkl --wrd_dim 100 --sentence_length 125 --restore checkpoint/model_max.ckpt
```

- inference(bucket)
```
$ python inference.py --mode bucket --emb_path embeddings/glove.6B.100d.txt.pkl --wrd_dim 100 --sentence_length 125 --restore checkpoint/model_max.ckpt < data/test.txt > pred.txt
$ python token_eval.py < pred.txt
$ python chunk_eval.py < pred.txt
```

- inference(line)
```
$ python inference.py --mode line --emb_path embeddings/glove.6B.100d.txt.pkl --wrd_dim 100 --sentence_length 125 --restore checkpoint/model_max.ckpt
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
D.C. NNP O B-GPE B-LOC

The Beatles were an English rock band formed in Liverpool in 1960.

The DT O O O
Beatles NNPS O B-PERSON B-MISC
were VBD O O O
an DT O O O
English JJ O B-LANGUAGE B-MISC
rock NN O O O
band NN O O O
formed VBN O O O
in IN O O O
Liverpool NNP O B-GPE B-LOC
in IN O O O
1960 CD O B-DATE O
. . O I-DATE O
```

### references

- general
  - articles
    - [Named Entity Recognition with Bidirectional LSTM-CNNs](https://www.aclweb.org/anthology/Q16-1026)
    - [Towards Deep Learning in Hindi NER: An approach to tackle the Labelled Data Scarcity](https://arxiv.org/pdf/1610.09756.pdf)
    - [Exploring neural architectures for NER](https://web.stanford.edu/class/cs224n/reports/6896582.pdf)
    - [Early Stopping(in Korean)](http://forensics.tistory.com/29)
    - [Learning Rate Decay](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay)
  - tensorflow impl
    - [ner-lstm](https://github.com/monikkinom/ner-lstm)
  - keras impl
    - [keras implementation](https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs)

- character convolution
  - articles
    - [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
    - [Implementing a sentence classification using Char level CNN & RNN](https://github.com/cuteboydot/Sentence-Classification-using-Char-CNN-and-RNN)
  - tensorflow impl
    - [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py)
    - [lstm-char-cnn-tensorflow/models/LSTMTDNN.py](https://github.com/carpedm20/lstm-char-cnn-tensorflow/blob/master/models/LSTMTDNN.py)

- multi-head attention
  - tensorflow impl
    - [transformer/modules.py](https://github.com/Kyubyong/transformer/blob/master/modules.py)
    - [transformer-tensorflow/transformer/attention.py](https://github.com/DongjunLee/transformer-tensorflow/blob/master/transformer/attention.py)

- CRF
  - articles
    - [Sequence Tagging with Tensorflow](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)
    - [ADVANCED: MAKING DYNAMIC DECISIONS AND THE BI-LSTM CRF](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)
  - tensorflow impl
    - [sequence_tagging](https://github.com/guillaumegenthial/sequence_tagging/blob/master/model/ner_model.py)
    - [tf_ner](https://github.com/guillaumegenthial/tf_ner)
    - [tensorflow/contrib/python/opps/crf.py](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/crf/python/ops/crf.py)

- ELMO
  - articles
    - [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)
  - tensorflow impl
    - [bilm](https://github.com/allenai/bilm-tf)
