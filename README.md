etagger
====

### description

- named entity tagger using multi-layer Bidirectional LSTM
  - + character convolutional embedding
  - + gazetteer features
  - + pos embedding
  - + multi-head attention

- base repository
  - https://github.com/monikkinom/ner-lstm
  - https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py
  - https://github.com/DongjunLee/transformer-tensorflow/blob/master/transformer/attention.py

- modification
  - modified for tf version(1.4)
  - removed unnecessary files
  - fixed bugs for MultiRNNCell()
  - refactoring
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
    - apply gazetteer features [done]
      - if we construct the gazetteer vocab from the training data, the performance is decreasing.
      - recommend constructing the gazetteer vocab from other sources.
    - apply learning-rate warm up [done]
    - apply pos embedding [done]
      - pos one-hot vector + pos embedding
    - extend language specific features [done]
      - initialCaps, allCaps, lowercase, mixedCaps, non-info
    - apply multi-head self-attention [doing]
      - without learning-rate warm up
    - apply ELMO embedding
    - serve api
      - freeze model and serve

- model
  ![graph-1](https://raw.githubusercontent.com/dsindex/etagger/master/etc/graph-1.png)

- references
  - [Named Entity Recognition with Bidirectional LSTM-CNNs](https://www.aclweb.org/anthology/Q16-1026)
    - [keras implementation](https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs)
  - [Towards Deep Learning in Hindi NER: An approach to tackle the Labelled Data Scarcity](https://arxiv.org/pdf/1610.09756.pdf)
  - [Exploring neural architectures for NER](https://web.stanford.edu/class/cs224n/reports/6896582.pdf)
  - character convolution
    - [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
    - [Implementing a sentence classification using Char level CNN & RNN](https://github.com/cuteboydot/Sentence-Classification-using-Char-CNN-and-RNN)
    - [lstm-char-cnn-tensorflow/models/LSTMTDNN.py](https://github.com/carpedm20/lstm-char-cnn-tensorflow/blob/master/models/LSTMTDNN.py)
  - multi-head attention
    - [transformer-tensorflow/transformer/attention.py](https://github.com/DongjunLee/transformer-tensorflow/blob/master/transformer/attention.py)
    - [transformer/modules.py](https://github.com/Kyubyong/transformer/blob/master/modules.py)

### pre-requisites

- tensorflow >= 1.4

- numpy

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
![graph-2](https://raw.githubusercontent.com/dsindex/etagger/master/etc/graph-2.png)

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

### experiments

- [README_DEV.md](https://github.com/dsindex/etagger/blob/master/README_DEV.md) 
