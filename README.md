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
    - apply gazetter features
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
$ tensorboard --logdir runs/summaries/
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
* weak points : B-ORG, I-ORG, B-MISC, I-MISC

* chr_embedding : max
precision:
I,0.885637974991
I-LOC,0.795454545455
B-ORG,0.864928909953
O,0.994457310885
I-PER,0.956375838926
I-MISC,0.591439688716
B-MISC,0.793696275072
I-ORG,0.857843137255
B-LOC,0.91
B-PER,0.944512946979

recall:
I,0.899285009862
I-LOC,0.817120622568
B-ORG,0.878988561108
O,0.991233075686
I-PER,0.98615916955
I-MISC,0.703703703704
B-MISC,0.789173789174
I-ORG,0.838323353293
B-LOC,0.927458033573
B-PER,0.947433518862

fscore:
I,0.892409321671
I-LOC,0.806142034549
B-ORG,0.871902060317
O,0.992842575634
I-PER,0.971039182283
I-MISC,0.642706131078
B-MISC,0.791428571429
I-ORG,0.847970926711
B-LOC,0.91864608076
B-PER,0.945970978697

total fscore:
0.892409321671

* chr embedding : conv
precision:
I,0.890855817361
I-LOC,0.824902723735
B-ORG,0.859597156398
O,0.993658219623
I-PER,0.970865467009
I-MISC,0.64
B-MISC,0.80394922426
I-ORG,0.851715686275
B-LOC,0.913813459268
B-PER,0.949068322981

recall:
I,0.899531558185
I-LOC,0.824902723735
B-ORG,0.873570138471
O,0.991622140375
I-PER,0.980103806228
I-MISC,0.740740740741
B-MISC,0.811965811966
I-ORG,0.832335329341
B-LOC,0.928057553957
B-PER,0.944959802103

fscore:
I,0.895172667607
I-LOC,0.824902723735
B-ORG,0.866527321589
O,0.99263913591
I-PER,0.975462763668
I-MISC,0.68669527897
B-MISC,0.807937632884
I-ORG,0.84191399152
B-LOC,0.920880428316
B-PER,0.947009606446

total fscore:
0.895172667607
```
