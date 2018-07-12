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
    - extend 5 class to 9 class [done]
    - apply dropout for train() only [done]
    - apply embedding_lookup()
      - word embedding [done]
    - apply character-level embedding
    - apply gazetter features
    - apply self-attention
    - apply ELMO embedding
    - serve api

### pre-requisites

- data
  - [download](https://github.com/mxhofer/Named-Entity-Recognition-BidirectionalLSTM-CNN-CoNLL/tree/master/data) 
  - place train.txt, dev.txt, test.txt in data dir

- glove embedding
  - [download](http://nlp.stanford.edu/data/glove.6B.zip)
  - unzip to 'embeddings' dir

### how to 

- convert word embedding to pickle
```
$ python embvec.py --emb_path embeddings/glove.6B.50d.txt --emb_dim 50
```

- check max sentence length
```
$ python check_sentence_length.py
train, max_sentence_length = 113
dev, max_sentence_length = 109
test, max_sentence_length = 124
```

- train
```
$ python train.py --emb_path embeddings/glove.6B.50d.txt.pkl --emb_dim 50 --sentence_length 125 --class_size 9
...
dev score:
precision, recall, fscore
[0.95510422234099412, 0.97158218125960061, 0.94342672413793105, 0.9135802469135802, 0.90273556231003038, 0.91424418604651159, 0.89659090909090911, 0.82258064516129037, 0.99417795509370943, 0.93149066855657925]
[0.97014115092290987, 0.96786534047436878, 0.95318454001088737, 0.86381322957198448, 0.88590604026845643, 0.83754993342210382, 0.85574837310195229, 0.73699421965317924, 0.99734729493891794, 0.91665698012321284]
[0.96256396444923231, 0.96972019931008058, 0.94828053073382079, 0.88800000000000001, 0.89424162589386524, 0.87421820708825571, 0.87569367369589346, 0.77743902439024393, 0.99576010315146302, 0.92401429492061637]
...
```

- inference(bulk)
```
$ python inference.py --emb_path embeddings/glove.6B.50d.txt.pkl --emb_dim 50 --sentence_length 125 --class_size 9 --restore checkpoint/model_max.ckpt
...
test score:
precision, recall, fscore
[0.95088161209068012, 0.97812773403324582, 0.88720930232558137, 0.75257731958762886, 0.8555018137847642, 0.83576642335766427, 0.74899328859060399, 0.62612612612612617, 0.99376315584314334, 0.87623701893708006]
[0.9338280766852195, 0.96712802768166095, 0.9148681055155875, 0.85214007782101164, 0.85189644792293795, 0.82275449101796405, 0.79487179487179482, 0.64351851851851849, 0.99188151683353221, 0.88412228796844183]
[0.94227769110764426, 0.97259678120922144, 0.90082644628099162, 0.79927007299270059, 0.85369532428355954, 0.82920941460470732, 0.7712508638562543, 0.63470319634703187, 0.99282144479781931, 0.88016199300484754]
total fscore:
0.880161993005
```

- inference(interactive)
```
$ python inference.py --interactive 1 --emb_path embeddings/glove.6B.50d.txt.pkl --emb_dim 50 --sentence_length 125 --class_size 9 --restore checkpoint/model_max.ckpt < data/test.txt > pred.txt
$ python eval.py < pred.txt
...
{'I': 7172, 'I-LOC': 219, 'B-ORG': 1415, 'O': 38241, 'B-PER': 1510, 'I-PER': 1118, 'I-MISC': 139, 'B-MISC': 558, 'I-ORG': 687, 'B-LOC': 1526}
{'I': 1013, 'I-LOC': 72, 'B-ORG': 239, 'O': 240, 'B-PER': 78, 'I-PER': 25, 'I-MISC': 83, 'B-MISC': 187, 'I-ORG': 135, 'B-LOC': 194}
{'I': 940, 'I-LOC': 38, 'B-ORG': 246, 'I-PER': 38, 'O': 313, 'I-MISC': 77, 'B-MISC': 144, 'I-ORG': 148, 'B-LOC': 142, 'B-PER': 107}

precision:
I,0.876237018937
I-LOC,0.752577319588
B-ORG,0.855501813785
O,0.993763155843
I-PER,0.978127734033
I-MISC,0.626126126126
B-MISC,0.748993288591
I-ORG,0.835766423358
B-LOC,0.887209302326
B-PER,0.950881612091

recall:
I,0.884122287968
I-LOC,0.852140077821
B-ORG,0.851896447923
O,0.991881516834
I-PER,0.967128027682
I-MISC,0.643518518519
B-MISC,0.794871794872
I-ORG,0.822754491018
B-LOC,0.914868105516
B-PER,0.933828076685

fscore:
I,0.880161993005
I-LOC,0.799270072993
B-ORG,0.853695324284
O,0.992821444798
I-PER,0.972596781209
I-MISC,0.634703196347
B-MISC,0.771250863856
I-ORG,0.829209414605
B-LOC,0.900826446281
B-PER,0.942277691108

total fscore:
0.880161993005
```
