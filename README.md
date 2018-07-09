etagger
====

### description

- named entity tagger using multi-layer Bidirectional LSTM

- original git repository
  - https://github.com/monikkinom/ner-lstm

- modification
  - modified for tf version(1.4)
  - removed unnecessary files
  - fix bugs for MultiRNNCell()
  - refactoring .... ing
    - implement input.py [done]
    - split model.py to model.py, train.py, inference.py [done]
    - extend 5 class to 9 class [done]
    - use embedding_lookup()
    - add character-level embedding
    - add gazetter features
    - add self-attention
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
precision, recall, fscore
[0.95392953929539293, 0.97600619195046434, 0.93837837837837834, 0.84701492537313428, 0.86409883720930236, 0.90261627906976749, 0.88022598870056501, 0.82802547770700641, 0.99444960520204362, 0.91958206151678801]
[0.9554831704668838, 0.96480489671002301, 0.94501905280348397, 0.88326848249027234, 0.88665175242356453, 0.82689747003994674, 0.84490238611713664, 0.75144508670520227, 0.99641652123327518, 0.91049633848657441]
[0.95470572280987254, 0.97037322046941132, 0.94168700840791986, 0.86476190476190473, 0.87523003312476999, 0.86309937456567065, 0.86220254565578314, 0.78787878787878785, 0.99543209159063173, 0.91501664622393564]
```

- inference(bulk)
```
$ python inference.py --emb_path embeddings/glove.6B.50d.txt.pkl --emb_dim 50 --sentence_length 125 --class_size 9 --restore checkpoint/model_max.ckpt
```

- inference(interactive)
```
$ python inference.py --interactive 1 --emb_path embeddings/glove.6B.50d.txt.pkl --emb_dim 50 --sentence_length 125 --class_size 9 --restore checkpoint/model_max.ckpt < data/test.txt > pred.txt
$ python eval.py < pred.txt
...
{'I-LOC': 215, 'B-ORG': 1415, 'O': 38276, 'B-PER': 1500, 'I-PER': 1111, 'I-MISC': 144, 'B-MISC': 535, 'I-ORG': 669, 'B-LOC': 1528}
{'I-LOC': 73, 'B-ORG': 237, 'O': 303, 'B-PER': 98, 'I-PER': 38, 'I-MISC': 94, 'B-MISC': 143, 'I-ORG': 113, 'B-LOC': 174}
{'I-LOC': 42, 'B-ORG': 246, 'I-PER': 45, 'O': 278, 'I-MISC': 72, 'B-MISC': 167, 'I-ORG': 166, 'B-LOC': 140, 'B-PER': 117}

precision:
I-LOC,0.746527777778
B-ORG,0.856537530266
O,0.992145986158
I-PER,0.966927763272
I-MISC,0.605042016807
B-MISC,0.789085545723
I-ORG,0.855498721228
B-LOC,0.89776733255
B-PER,0.938673341677

recall:
I-LOC,0.836575875486
B-ORG,0.851896447923
O,0.99278933444
I-PER,0.96107266436
I-MISC,0.666666666667
B-MISC,0.762108262108
I-ORG,0.80119760479
B-LOC,0.916067146283
B-PER,0.927643784787

fscore:
I-LOC,0.788990825688
B-ORG,0.85421068518
O,0.99246755604
I-PER,0.96399132321
I-MISC,0.63436123348
B-MISC,0.775362318841
I-ORG,0.82745825603
B-LOC,0.906824925816
B-PER,0.933125972006
```
