etagger
====

### description

- original git repository
  - https://github.com/monikkinom/ner-lstm

- modification
  - modified for tf version(1.4)
  - removed unnecessary files
  - fix bugs for MultiRNNCell()
  - refactoring .... ing
    - split model.py, train.py, inference.py
    - remove get_conll_embeddings.py and use embedding lookup

### pre-requisites

- data
  - [download](https://github.com/mxhofer/Named-Entity-Recognition-BidirectionalLSTM-CNN-CoNLL) 
  - place train.txt, dev.txt, test.txt in data dir

- glove embedding
  - [download](http://nlp.stanford.edu/data/glove.6B.zip)
  - unzip to 'embedding' dir

### how to 

- make glove pickle data
```
$ cd embeddings
$ python glove_model.py --dimension 50 --restore glove.6B.50d.txt
```

- convert train/dev/test text file to embedding format
```
$ cd embeddings
$ python get_conll_embeddings.py --train ../data/train.txt --test_a ../data/dev.txt --test_b ../data/test.txt --use_model glovevec_model_50.pkl --model_dim 50 --sentence_length 125

* for small sample data
$ python get_conll_embeddings.py --train ../data/eng.train_50 --test_a ../data/eng.test_a_50 --test_b ../data/eng.test_b_50 --use_model glovevec_model_50.pkl --model_dim 50 --sentence_length 125
```

- train
```
$ python model.py --word_dim 61 --sentence_length 125 --class_size 5 --rnn_size 256 --num_layers 1 --batch_size 128 --epoch 50
...

(num_layers 1)
test_a score:
[0.96773173046504268, 0.93732327992459941, 0.88112566715186802, 0.8534119629317608, 0.99481961576881084, 0.92333841284726292]

: PER, LOC, ORG, MISC, O

(num_layers 2)
test_a score:
[0.96857142857142853, 0.93333333333333335, 0.88687561214495603, 0.8617754357519255, 0.9952267580279891, 0.92487244149903236]
```
