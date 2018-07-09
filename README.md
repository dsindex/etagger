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
    - split model.py, train.py, inference.py

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
$ python train.py --emb_path embeddings/glove.6B.50d.txt.pkl --emb_dim 50 --sentence_length 125 --class_size 5 
```

- inference(bulk)
```
$ python inference.py --emb_path embeddings/glove.6B.50d.txt.pkl --emb_dim 50 --sentence_length 125 --class_size 5 --restore checkpoint/model_max.ckpt
...
precision, recall, fscore
[0.96745886654478974, 0.88107287449392713, 0.84672677381419048, 0.73208722741433019, 0.99362659660258579, 0.88170212765957445]
[0.95420122610890734, 0.90441558441558445, 0.86538461538461542, 0.76797385620915037, 0.99071432276806559, 0.89398422090729779]
[0.96078431372549022, 0.89259164316841832, 0.85595403209827625, 0.74960127591706538, 0.99216832261835186, 0.88780069780253423]
```

- inference(interactive)
```
$ python inference.py --interactive 1 --emb_path embeddings/glove.6B.50d.txt.pkl --emb_dim 50 --sentence_length 125 --class_size 5 --restore checkpoint/model_max.ckpt < data/test.txt > pred.txt
$ python eval.py < pred.txt
{'LOC': 1740, 'MISC': 698, 'PER': 2641, 'O': 38192, 'ORG': 2157}
{'LOC': 241, 'MISC': 266, 'PER': 102, 'O': 245, 'ORG': 384}
{'LOC': 185, 'MISC': 220, 'PER': 132, 'O': 362, 'ORG': 339}

precision:
LOC,0.87834427057
MISC,0.724066390041
O,0.993625933345
PER,0.962814436748
ORG,0.848878394333

recall:
LOC,0.903896103896
MISC,0.760348583878
O,0.990610572184
PER,0.952398124775
ORG,0.864182692308

fscore:
LOC,0.890937019969
MISC,0.741764080765
O,0.992115961606
PER,0.95757795504
ORG,0.856462179869
```
