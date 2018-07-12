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
    - implement input.py [done]
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
precision, recall, fscore
[0.95392953929539293, 0.97600619195046434, 0.93837837837837834, 0.84701492537313428, 0.86409883720930236, 0.90261627906976749, 0.88022598870056501, 0.82802547770700641, 0.99444960520204362, 0.91958206151678801]
[0.9554831704668838, 0.96480489671002301, 0.94501905280348397, 0.88326848249027234, 0.88665175242356453, 0.82689747003994674, 0.84490238611713664, 0.75144508670520227, 0.99641652123327518, 0.91049633848657441]
[0.95470572280987254, 0.97037322046941132, 0.94168700840791986, 0.86476190476190473, 0.87523003312476999, 0.86309937456567065, 0.86220254565578314, 0.78787878787878785, 0.99543209159063173, 0.91501664622393564]
...
```

- inference(bulk)
```
$ python inference.py --emb_path embeddings/glove.6B.50d.txt.pkl --emb_dim 50 --sentence_length 125 --class_size 9 --restore checkpoint/model_max.ckpt
...
test score:
precision, recall, fscore
[0.95502810743285449, 0.96189669771380182, 0.91376255168340226, 0.76254180602006694, 0.87401095556908093, 0.85819975339087551, 0.79858156028368799, 0.5864661654135338, 0.99430680843320252, 0.88925478716916695]
[0.94557823129251706, 0.98269896193771622, 0.92745803357314149, 0.88715953307392992, 0.86453943407585787, 0.83353293413173657, 0.80199430199430199, 0.72222222222222221, 0.99206308035482704, 0.89879191321499019]
[0.95027967681789938, 0.97218656397090286, 0.92055935733412675, 0.82014388489208645, 0.86924939467312345, 0.84568651275820172, 0.80028429282160629, 0.64730290456431538, 0.99318367717895117, 0.8939979155171357]
total fscore:
0.893997915517
```

- inference(interactive)
```
$ python inference.py --interactive 1 --emb_path embeddings/glove.6B.50d.txt.pkl --emb_dim 50 --sentence_length 125 --class_size 9 --restore checkpoint/model_max.ckpt < data/test.txt > pred.txt
$ python eval.py < pred.txt
...
{'I': 7291, 'I-LOC': 228, 'B-ORG': 1436, 'O': 38248, 'I-PER': 1136, 'I-MISC': 156, 'B-MISC': 563, 'I-ORG': 696, 'B-LOC': 1547, 'B-PER': 1529}
{'I': 908, 'I-LOC': 71, 'B-ORG': 207, 'O': 219, 'I-PER': 45, 'I-MISC': 110, 'B-MISC': 142, 'I-ORG': 115, 'B-LOC': 146, 'B-PER': 72}
{'I': 821, 'I-LOC': 29, 'B-ORG': 225, 'I-PER': 20, 'O': 306, 'I-MISC': 60, 'B-MISC': 139, 'I-ORG': 139, 'B-LOC': 121, 'B-PER': 88}

precision:
I,0.889254787169
I-LOC,0.76254180602
B-ORG,0.874010955569
O,0.994306808433
I-PER,0.961896697714
I-MISC,0.586466165414
B-MISC,0.798581560284
I-ORG,0.858199753391
B-LOC,0.913762551683
B-PER,0.955028107433

recall:
I,0.898791913215
I-LOC,0.887159533074
B-ORG,0.864539434076
O,0.992063080355
I-PER,0.982698961938
I-MISC,0.722222222222
B-MISC,0.801994301994
I-ORG,0.833532934132
B-LOC,0.927458033573
B-PER,0.945578231293

fscore:
I,0.893997915517
I-LOC,0.820143884892
B-ORG,0.869249394673
O,0.993183677179
I-PER,0.972186563971
I-MISC,0.647302904564
B-MISC,0.800284292822
I-ORG,0.845686512758
B-LOC,0.920559357334
B-PER,0.950279676818

total fscore:
0.893997915517
```
