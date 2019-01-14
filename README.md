# ETagger: Entity Tagger

## Description

- personally, i'm interested in NER tasks. so, i decided to implement a sequence tagging model which consists of 
  - encoding
    - basic embedding
      - [x] 1) word embedding(glove) and character convolutional embedding
      - [x] 2) ELMo embedding
      - [ ] 3) BERT embedding
    - etc embedding
      - [x] pos embedding, etc features(shape, pos, gazetteer(not used))
  - contextual encoding
    - [x] 1) multi-layer BiLSTM
    - [x] 2) Transformer(encoder)
  - decoding
    - [x] CRF decoder

- there are so many repositories available for reference. i borrowed those codes as many as possible to use here.
  - [ner-lstm](https://github.com/monikkinom/ner-lstm)
  - [cnn-text-classification-tf/text_cnn.py](https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py)
  - [transformer/modules.py](https://github.com/Kyubyong/transformer/blob/master/modules.py)
  - [sequence_tagging/ner_model.py](https://github.com/guillaumegenthial/sequence_tagging/blob/master/model/ner_model.py)
  - [tf_ner/masked_conv.py](https://github.com/guillaumegenthial/tf_ner/blob/master/models/chars_conv_lstm_crf/masked_conv.py)
  - [torchnlp/layers.py](https://github.com/kolloldas/torchnlp/blob/master/torchnlp/modules/transformer/layers.py)
  - [bilm-tf](https://github.com/allenai/bilm-tf)
  - [tensorflow-cmake](https://github.com/PatWie/tensorflow-cmake)
  - [medium-tffreeze-1.py](https://gist.github.com/morgangiraud/249505f540a5e53a48b0c1a869d370bf#file-medium-tffreeze-1-py)
  - [medium-tffreeze-2.py](https://gist.github.com/morgangiraud/5ef49adc3c608bf639164b1dd5ed3dab#file-medium-tffreeze-2-py)

- my main questions are :
  - can this module perform at the level of state of the art?
    - [x] the f1 score is near SOTA.
      - 0.92536
  - how to make it faster when it comes to using the BiLSTM?
    - [x] the solution is LSTMBlockFusedCell().
      - 3.13 times faster than LSTMCell() during training time.
      - 1.26 times faster than LSTMCell() during inference time.
  - can the Transformer have competing results against the BiLSTM? and how much faster?
    - [x] contextual encoding by the Transformer encoder yields competing results.
      - in case the sequence to sequence model like translation, the multi-head attention mechanism might be very powerful for alignments.
      - however, for sequence tagging, the source of power is from point-wise feed forward net with wide range of kernel size. it is not from the multi-head attention only.
        - if you are using kernel size 1, then the the performance will be very worse than you expect.
      - it seems that point-wise feed forward net collects contextual information in the layer by layer manner.
        - this is very similar with hierarchical convolutional neural network.
      - i'd like to say `Attention is Not All you need`
    - [x] you can see the below evaluation results.
      - multi-layer BiLSTM using LSTMBlockFusedCell() is slightly faster than the Transformer with 4 layers on GPU.
      - moreover, the BiLSTM is 2 times faster on CPU environment(multi-thread) than on GPU.
        - LSTMBlockFusedCell() is well optimized for multi-core CPU via multi-threading.
        - i guess there might be an overhead when copying b/w GPU memory and main memory.
      - the BiLSTM is 3 ~ 4 times faster than the Transformer version on 1 CPU(single-thread)
      - during inference time, 1 layer BiLSTM on 1 CPU takes just **5.1 msec** per sentence on average.
  - how to use a trained model from C++? is it much faster?
    - [x] save the meta graph and trained parameters. and restore it via tensorflow C++ API.
      - 1 layer BiLSTM on multi CPU takes **2.7 msec** per sentence on average.
      - 1 layer BiLSTM on single CPU takes **3.9 msec** per sentence on average.

## Models and Evaluation

- models
  - BiLSTM
  ![BiLSTM](https://raw.githubusercontent.com/dsindex/etagger/master/etc/graph-1-1.png)
  - Transformer
  ![Transformer](https://raw.githubusercontent.com/dsindex/etagger/master/etc/graph-1-2.png)

- evaluation
  - [experiment logs](https://github.com/dsindex/etagger/blob/master/README_DEV.md)
  - results
    - Transformer
      - Glove
        - setting
          - `experiments 7, test 9`
          - rnn_type : fused, rnn_used : False, tf_used : True, tf_num_layers : 4
        - per-token(partial) f1 : 0.9083215796897038
        - per-chunk(exact)   f1 : **0.904078014184397**
        - average processing time per bucket
          - 1 GPU(TITAN X (Pascal), 12196MiB)
            - restore version        : 0.013825567226844812 sec
            - frozen version         : 0.015376264122228799 sec
            - tensorRT(FP16) version : no meaningful difference
          - 32 core CPU(multi-threading)
            - python : 0.017238136546748987 sec
            - C++ : 0.013 sec
          - 1 CPU(single-thread)
            - python : 0.03358284470571628 sec
            - C++ : 0.021510 sec
    - BiLSTM
      - Glove
        - setting
          - `experiments 9, test 1`
          - rnn_type : fused, rnn_used : True, rnn_num_layers : 2, tf_used : False
        - per-token(partial) f1 : 0.9152852267186738
        - per-chunk(exact)   f1 : **0.9094911075893644**
        - average processing time per bucket
          - 1 GPU(TITAN X (Pascal), 12196MiB)
            - restore version        : 0.010454932072004718 sec
            - frozen version         : 0.011339560587942018 sec
            - tensorRT(FP16) version : no meaningful difference
          - 32 core CPU(multi-threading)
            - rnn_num_layers 2 : 0.006132203450549827 sec
            - rnn_num_layers 1
              - python : 0.0041805055967241884 sec
              - C++
                - 0.002735 sec
                - 0.002175 sec (`experiments 9, test 2`), 0.8800
                - 0.002783 sec (`experiments 9, test 3`), 0.8858
                - 0.004407 sec (`experiments 9, test 4`), 0.8887
                - 0.003687 sec (`experiments 9, test 5`), 0.8835
                - 0.002976 sec (`experiments 9, test 6`), 0.8782
                - 0.002855 sec (`experiments 9, test 7`), 0.8906
                  - 0.002697 sec with optimizations for FMA, AVX and SSE. no meaningful difference.
          - 1 CPU(single-thread)
            - rnn_num_layers 2 : 0.008001159379070668 sec 
            - rnn_num_layers 1
              - python : 0.0051817628640952506 sec
              - C++
                - 0.003998 sec
                - 0.002853 sec (`experiments 9, test 2`)
                - 0.003474 sec (`experiments 9, test 3`)
                - 0.005118 sec (`experiments 9, test 4`)
                - 0.004139 sec (`experiments 9, test 5`)
                - 0.004133 sec (`experiments 9, test 6`)
                - 0.003334 sec (`experiments 9, test 7`)
                  - 0.003078 sec with optimizations for FMA, AVX and SSE. no meaningful difference.
      - ELMo
        - setting
          - `experiments 8, test 2`
          - rnn_type : fused, rnn_used : True, rnn_num_layers : 2, tf_used : False
        - per-token(partial) f1 : 0.9322728663199756
        - per-chunk(exact)   f1 : **0.9253625751680227**
        ```
        $ etc/conlleval < pred.txt
        processed 46666 tokens with 5648 phrases; found: 5662 phrases; correct: 5234.
        accuracy:  98.44%; precision:  92.44%; recall:  92.67%; FB1:  92.56
                      LOC: precision:  94.29%; recall:  92.99%; FB1:  93.63  1645
                     MISC: precision:  84.38%; recall:  84.62%; FB1:  84.50  704
                      ORG: precision:  89.43%; recall:  91.69%; FB1:  90.55  1703
                      PER: precision:  97.27%; recall:  96.85%; FB1:  97.06  1610
        ```
        - average processing time per bucket
          - 1 GPU(TITAN X (Pascal), 12196MiB) : 0.06133532517637155 sec -> need to recompute
          - 1 GPU(Tesla V100)                 : 0.029950057644797457 sec
          - 32 core CPU(multi-threading)      : 0.40098162731570347 sec
          - 1 CPU(single-thread)              : 0.7398052649182165 sec
      - ELMo + Glove
        - setting
          - `experiments 10, test 5`
          - rnn_type : fused, rnn_used : True, rnn_num_layers : 2, tf_used : False
        - per-token(partial) f1 : 0.9280469897209985
        - per-chunk(exact)   f1 : 0.9212688875143589
        ```
        processed 46666 tokens with 5648 phrases; found: 5675 phrases; correct: 5214.
        accuracy:  98.34%; precision:  91.88%; recall:  92.32%; FB1:  92.10
              LOC: precision:  93.32%; recall:  93.82%; FB1:  93.57  1677
             MISC: precision:  81.78%; recall:  83.76%; FB1:  82.76  719
              ORG: precision:  89.51%; recall:  89.89%; FB1:  89.70  1668
              PER: precision:  97.33%; recall:  96.97%; FB1:  97.15  1611
        ```
        - average processing time per bucket
          - 1 GPU(TITAN X (Pascal), 12196MiB) : 0.036233977567360014 sec
    - BiLSTM + Transformer
      - Glove
        - setting
          - `experiments 7, test 10`
          - rnn_type : fused, rnn_used : True, rnn_num_layers : 2, tf_used : True, tf_num_layers : 1
        - per-token(partial) f1 : 0.910979409787988
        - per-chunk(exact)   f1 : **0.9047451049567825**
    - BiLSTM + multi-head attention
      - Glove
        - setting
          - `experiments 6, test 7`
        - per-token(partial) f1 : 0.9157317073170732
        - per-chunk(exact)   f1 : **0.9102156238953694**
  - comparision to previous research
    - implementations
      - [Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs](https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs)
        - tested
        - Glove6B.100
        - Prec: 0.887, Rec: 0.902, F1: 0.894
      - [sequence_tagging](https://github.com/guillaumegenthial/sequence_tagging)
        - tested
        - Glove6B.100
        - F1: 0.8998
      - [tf_ner](https://github.com/guillaumegenthial/tf_ner)
        - tested
        - Glove840B.300
        - F1 : 0.905 ~ 0.907 (chars_conv_lstm_crf)
          - reported F1 : 0.9118
      - [torchnlp](https://github.com/kolloldas/torchnlp)
        - tested
        - Glove6B.200
        - F1 : 0.8845
          - just 1 block of Transformer encoder
    - SOTA
      - [Contextual String Embeddings for Sequence Labeling](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view)
        - reported F1 : 0.9309
      - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
        - reported F1 : 0.928
      - [Semi-Supervised Sequence Modeling with Cross-View Training](https://arxiv.org/pdf/1809.08370.pdf)
        - reported F1 : 0.926
      - [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)
        - reported F1 : 0.9222
      - [Semi-supervised sequence tagging with bidirectional language models](https://arxiv.org/pdf/1705.00108.pdf)
        - reported F1 : 0.9193

## Pre-requisites

- python >= 3.6

- tensorflow >= 1.10
```
$ python -m pip install tensorflow-gpu
* version matches
  tensorflow 1.10, CUDA 9.0, cuDNN 7.12
  (cuda-9.0-pkg/cuda-9.0/lib64, cudnn-9.0/lib64)
  tensorflow 1.11, CUDA 9.0, cuDNN 7.31
  (cuda-9.0-pkg/cuda-9.0/lib64, cudnn-9.0-v73/lib64)
  tensorflow 1.11, CUDA 9.0, cuDNN 7.31, TensorRT 4.0
  (cuda-9.0-pkg/cuda-9.0/lib64, cudnn-9.0-v73/lib64, TensorRT-4.0.1.6/lib)
```

- numpy

- data
  - merge train.txt, dev.txt, test.txt to total.txt in data dir

- glove embedding
  - [download Glove6B](http://nlp.stanford.edu/data/glove.6B.zip)
  - [download Glove840B](http://nlp.stanford.edu/data/glove.840B.300d.zip)
  - unzip to 'embeddings' dir

- bilm
  - install [bilm-tf](https://github.com/allenai/bilm-tf)
  - download [ELMo weights and options](https://allennlp.org/elmo)
  ```
  $ ls embeddings
  embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json  embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
  ``` 
  - test
  ```
  * after run embvec.py
  $ python test_bilm.py
  ```

- spacy [optional]
  - if you want to analyze input string and see how it detects entities, then you need to install spacy lib.
  ```
  $ pip install spacy
  $ python -m spacy download en
  ```

## How to run

- convert word embedding to pickle
```
* for Glove
$ python embvec.py --emb_path embeddings/glove.6B.50d.txt  --wrd_dim 50  --train_path data/train.txt --total_path data/total.txt > embeddings/vocab.txt
$ python embvec.py --emb_path embeddings/glove.6B.100d.txt --wrd_dim 100 --train_path data/train.txt --total_path data/total.txt > embeddings/vocab.txt
$ python embvec.py --emb_path embeddings/glove.6B.200d.txt --wrd_dim 200 --train_path data/train.txt --total_path data/total.txt > embeddings/vocab.txt
$ python embvec.py --emb_path embeddings/glove.6B.300d.txt --wrd_dim 300 --train_path data/train.txt --total_path data/total.txt > embeddings/vocab.txt
$ python embvec.py --emb_path embeddings/glove.840B.300d.txt --wrd_dim 300 --train_path data/train.txt --total_path data/total.txt --lowercase 0 > embeddings/vocab.txt

* for ELMo
$ python embvec.py --emb_path embeddings/glove.6B.50d.txt --wrd_dim 50 --train_path data/train.txt --total_path data/total.txt --elmo_vocab_path embeddings/elmo_vocab.txt --elmo_options_path embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json --elmo_weight_path embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 > embeddings/vocab.txt
$ python embvec.py --emb_path embeddings/glove.6B.300d.txt --wrd_dim 300 --train_path data/train.txt --total_path data/total.txt --elmo_vocab_path embeddings/elmo_vocab.txt --elmo_options_path embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json --elmo_weight_path embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 > embeddings/vocab.txt
```

- train
```
* for Glove
$ python train.py --emb_path embeddings/glove.6B.100d.txt.pkl --wrd_dim 100 --batch_size 20 --epoch 70
$ python train.py --emb_path embeddings/glove.6B.300d.txt.pkl --wrd_dim 300 --batch_size 20 --epoch 70
$ python train.py --emb_path embeddings/glove.840B.300d.txt.pkl --wrd_dim 300 --batch_size 20 --epoch 70

* form ELMo
$ python train.py --emb_path embeddings/glove.6B.50d.txt.pkl --wrd_dim 50 --batch_size 20 --epoch 70
$ python train.py --emb_path embeddings/glove.6B.300d.txt.pkl --wrd_dim 300 --batch_size 20 --epoch 70

$ rm -rf runs; tensorboard --logdir runs/summaries/ --port 6008
```
    
- inference(bulk)
```
$ python inference.py --emb_path embeddings/glove.6B.100d.txt.pkl --wrd_dim 100 --restore checkpoint/ner_model
```

- inference(bucket)
```
* for Glove
$ python inference.py --mode bucket --emb_path embeddings/glove.6B.100d.txt.pkl --wrd_dim 100 --restore checkpoint/ner_model < data/test.txt > pred.txt
$ python inference.py --mode bucket --emb_path embeddings/glove.6B.300d.txt.pkl --wrd_dim 300 --restore checkpoint/ner_model < data/test.txt > pred.txt
$ python inference.py --mode bucket --emb_path embeddings/glove.840B.300d.txt.pkl --wrd_dim 300 --restore checkpoint/ner_model < data/test.txt > pred.txt

* for ELMo
$ python inference.py --mode bucket --emb_path embeddings/glove.6B.50d.txt.pkl --wrd_dim 50 --restore checkpoint/ner_model < data/test.txt > pred.txt
$ python inference.py --mode bucket --emb_path embeddings/glove.6B.300d.txt.pkl --wrd_dim 300 --restore checkpoint/ner_model < data/test.txt > pred.txt

$ python token_eval.py < pred.txt
$ python chunk_eval.py < pred.txt
```

- inference(line)
```
$ python inference.py --mode line --emb_path embeddings/glove.6B.100d.txt.pkl --wrd_dim 100 --restore checkpoint/ner_model
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

- inference(bucket) using frozen model, tensorRT, C++
  - [tensorflow-cmake](https://github.com/PatWie/tensorflow-cmake)
  - [build tensorflow from source](https://www.tensorflow.org/install/source)
  ```
  * create virtual env `python -m venv python3.6_tfsrc` and activate it.
  $ python -m vent python3.6_tfsrc
  $ source /home/python3.6_tfsrc/bin/activate

  * build tensorflow from source.
  $ git clone https://github.com/tensorflow/tensorflow.git tensorflow-src-cpu
  $ cd tensorflow-src-cpu
  * you should checkout the same version of pip used for training.
  $ git checkout r1.11
  * configure without CUDA
  $ ./configure

  * build pip package with optimizations for FMA, AVX and SSE( https://medium.com/@sometimescasey/building-tensorflow-from-source-for-sse-avx-fma-instructions-worth-the-effort-fbda4e30eec3 ).
  $ python -m pip install --upgrade pip
  $ python -m pip install --upgrade setuptools
  $ bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 //tensorflow/tools/pip_package:build_pip_package
  $ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
  * install pip package
  $ python -m pip install /tmp/tensorflow_pkg/tensorflow-1.11.0-cp36-cp36m-linux_x86_64.whl

  * build libraries we need.
  $ bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 //tensorflow:libtensorflow.so
  $ bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 //tensorflow:libtensorflow_cc.so

  * copy libraries and headers to dist directory.
  $ export TENSORFLOW_SOURCE_DIR='/home/tensorflow-src-cpu'
  $ export TENSORFLOW_BUILD_DIR='/home/tensorflow-dist-cpu'
  $ mkdir -p ${TENSORFLOW_BUILD_DIR}/includes/tensorflow/cc/ops
  $ cp -rf ${TENSORFLOW_SOURCE_DIR}/bazel-bin/tensorflow/*.so ${TENSORFLOW_BUILD_DIR}/
  $ cp -rf ${TENSORFLOW_SOURCE_DIR}/bazel-genfiles/tensorflow/cc/ops/*.h ${TENSORFLOW_BUILD_DIR}/includes/tensorflow/cc/ops/

  * for LSTMBlockFusedCell()
  $ rnn_path=`python -c "import tensorflow; print(tensorflow.contrib.rnn.__path__[0])"`
  $ rnn_ops_lib=${rnn_path}/python/ops/_lstm_ops.so
  $ cp -rf ${rnn_ops_lib} ${TENSORFLOW_BUILD_DIR}
  $ export LD_LIBRARY_PATH=${TENSORFLOW_BUILD_DIR}:$LD_LIBRARY_PATH
  ```
  - `.bashrc` sample
  ```
  # tensorflow so, header dist
  export TENSORFLOW_SOURCE_DIR='/home/tensorflow-src-cpu'
  export TENSORFLOW_BUILD_DIR='/home/tensorflow-dist-cpu'
  # for loading _lstm_ops.so
  export LD_LIBRARY_PATH=${TENSORFLOW_BUILD_DIR}:$LD_LIBRARY_PATH
  ```
  - *test* build sample model and inference by C++
  ```
  $ cd /home/etagger
  * build and save sample model
  $ cd inference
  $ python train_sample.py

  * inference using python
  $ python python/inference_sample.py

  * inference using c++
  * edit etagger/inference/cc/CMakeLists.txt
    find_package(TensorFlow 1.11 EXACT REQUIRED)
  $ cd etagger/inference/cc
  $ mkdir build
  $ cd build
  $ cmake ..
  $ make
  $ cd ../..
  $ ./cc/build/inference_sample
  ```
  - *test* build iris model, freezing and inference by C++
  ```
  $ cd /home/etagger
  * build and save iris model
  $ cd inference
  $ python train_iris.py

  * freeze graph
  $ python freeze.py --model_dir exported --output_node_names logits --frozen_model_name iris_frozen.pb

  * inference using python
  $ python python/inference_iris.py

  * inference using C++
  * edit etagger/inference/cc/CMakeLists.txt
    find_package(TensorFlow 1.11 EXACT REQUIRED)
  $ cd etagger/inference/cc
  $ mkdir build
  $ cd build
  $ cmake ..
  $ make
  $ cd ../..
  $ ./cc/build/inference_iris
  ```
  - export etagger model, freezing and inference by C++
  ```
  $ cd inference
  * let's assume that we have a saved model :
  *   1) BiLSTM, LSTMCell(), without ELMo
  *   2) BiLSTM, LSTMBlockFusedCell(), withoug ELMo
  *     : can't find `BlockLSTM` when using import_meta_graph()
  *     : similar issue => https://stackoverflow.com/questions/50298058/restore-trained-tensorflow-model-keyerror-blocklstm
        : how to fix? => https://github.com/tensorflow/tensorflow/issues/23369
        : what about C++? => https://stackoverflow.com/questions/50475320/executing-frozen-tensorflow-graph-that-uses-tensorflow-contrib-resampler-using-c
          we can load '_lstm_ops.so' for LSTMBlockFusedCell().
  *   3) Transformer, without ELMo

  * restore the model to check list of operations, placeholders and tensors for mapping. and export it another place.
  $ python export.py --restore ../checkpoint/ner_model --export exported/ner_model --export-pb exported

  * freeze graph
  $ python freeze.py --model_dir exported --output_node_names logits,loss/trans_params,sentence_lengths --frozen_model_name ner_frozen.pb

  * inference using python
  $ python python/inference.py --emb_path ../embeddings/glove.6B.100d.txt.pkl --wrd_dim 100 --frozen_path exported/ner_frozen.pb < ../data/test.txt > pred.txt
  $ python python/inference.py --emb_path ../embeddings/glove.6B.300d.txt.pkl --wrd_dim 300 --frozen_path exported/ner_frozen.pb < ../data/test.txt > pred.txt
  $ python python/inference.py --emb_path ../embeddings/glove.840B.300d.txt.pkl --wrd_dim 300 --frozen_path exported/ner_frozen.pb < ../data/test.txt > pred.txt

  * inference using python with optimized graph_def via tensorRT (only for GPU)
  $ python python/inference_trt.py --emb_path ../embeddings/glove.6B.100d.txt.pkl --wrd_dim 100 --frozen_path exported/ner_frozen.pb < ../data/test.txt > pred.txt
  $ python python/inference_trt.py --emb_path ../embeddings/glove.6B.300d.txt.pkl --wrd_dim 300 --frozen_path exported/ner_frozen.pb < ../data/test.txt > pred.txt
  $ python python/inference_trt.py --emb_path ../embeddings/glove.840B.300d.txt.pkl --wrd_dim 300 --frozen_path exported/ner_frozen.pb < ../data/test.txt > pred.txt
  * inspect `pred.txt` whether the predictions are same.
  $ python ../token_eval.py < pred.txt

  * inference using C++
  $ ./cc/build/inference exported/ner_frozen.pb ../embeddings/vocab.txt < ../data/test.txt > pred.txt
  * inspect `pred.txt` whether the predictions are same.
  $ python ../token_eval.py < pred.txt
  ```

## Development note

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

- train, dev accuracy after applying LSTMBlockFusedCell which is 2~3 times faster than LSTMCell
![graph-5](https://raw.githubusercontent.com/dsindex/etagger/master/etc/graph-5.png)

- tips for training speed up
  - filter out words(which are not in train/dev/test data) from word embeddings. but not for service.
  - use LSTMBlockFusedCell for bidirectional LSTM. this is slightly faster than LSTMCell.
    - about 3.13 times faster during training time.
      - 297.6699993610382 sec -> 94.96637988090515 sec for 1 epoch
    - about 1.26 times faster during inference time.
      - 0.010652577061606541 sec -> 0.008411417501886556 sec for 1 sentence
    - where is the LSTMBlockFusedCell() defined?
    ```
    https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/rnn/python/ops/lstm_ops.py
    vi ../lib/python3.6/site-packages/tensorflow/contrib/rnn/ops/gen_lstm_ops.py
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/ops/lstm_ops.cc
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/kernels/lstm_ops.cc
    ```
  - use early stopping

- tips for Transformer
  - start with small learning rate.
  - be careful to use residual connection after multi-head attention or feed forward net.
    - `x = tf.nn.dropout(x + y)` -> `x = tf.nn.dropout(x_norm + y)`
  - the f1 of train/dev by token are relatively lower than the f1 of the BiLSTM. but after applying the CRF layer, those f1 by token are increased very sharply.
    - does it mean that the Transformer is weak for collecting context for deciding label at the current position? then, how to overcome?
    - try to revise the position-wise feed forward net
      - padding before and after
        - (batch_size, sentence_length, model_dim) -> (batch_size, 1+sentence_length+1, model_dim)
      - conv1d with kernel size 1 -> 3
      - this is the key to sequence taggging problems.
    - after applying kernel_size 3
    ![graph-6](https://raw.githubusercontent.com/dsindex/etagger/master/etc/graph-6.png)

- tips in general
  - save best model by using token-based f1. token-based f1 is slightly better than chunk-based f1
  - be careful for word lowercase when you are using glove6B embeddings. those are all lowercased.
  - feed max sentence length to session. this yields huge improvement of inference speed.
  - when it comes to using import_meta_graph(), you should run global_variable_initialzer() before restore().

## References

- general
  - articles
    - [Named Entity Recognition with Bidirectional LSTM-CNNs](https://www.aclweb.org/anthology/Q16-1026)
    - [Towards Deep Learning in Hindi NER: An approach to tackle the Labelled Data Scarcity](https://arxiv.org/pdf/1610.09756.pdf)
    - [Exploring neural architectures for NER](https://web.stanford.edu/class/cs224n/reports/6896582.pdf)
    - [Early Stopping(in Korean)](http://forensics.tistory.com/29)
    - [Learning Rate Decay](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay)
  - tensorflow impl
    - [ner-lstm](https://github.com/monikkinom/ner-lstm)
    - [sequence_tagging](https://github.com/guillaumegenthial/sequence_tagging)
    - [tf_ner](https://github.com/guillaumegenthial/tf_ner)
  - keras impl
    - [Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs](https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs)
  - pytorch impl
    - [torchnlp](https://github.com/kolloldas/torchnlp/tree/master/torchnlp)

- character convolution
  - articles
    - [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
    - [Implementing a sentence classification using Char level CNN & RNN](https://github.com/cuteboydot/Sentence-Classification-using-Char-CNN-and-RNN)
  - tensorflow impl
    - [cnn-text-classification-tf/text_cnn.py](https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py)
    - [lstm-char-cnn-tensorflow/LSTMTDNN.py](https://github.com/carpedm20/lstm-char-cnn-tensorflow/blob/master/models/LSTMTDNN.py)

- Transformer
  - articles
    - [Building the Mighty Transformer for Sequence Tagging in PyTorch](https://medium.com/@kolloldas/building-the-mighty-transformer-for-sequence-tagging-in-pytorch-part-i-a1815655cd8)
    - [QANET: COMBINING LOCAL CONVOLUTION WITH GLOBAL SELF-ATTENTION FOR READING COMPREHENSION](https://arxiv.org/pdf/1804.09541.pdf)
  - tensorflow impl
    - [transformer/modules.py](https://github.com/Kyubyong/transformer/blob/master/modules.py)
    - [transformer-tensorflow/attention.py](https://github.com/DongjunLee/transformer-tensorflow/blob/master/transformer/attention.py)
    - [seq2seq/pooling_encoder.py](https://github.com/google/seq2seq/blob/master/seq2seq/encoders/pooling_encoder.py)
  - pytorch impl
    - [torchnlp/sublayers.py](https://github.com/kolloldas/torchnlp/blob/master/torchnlp/modules/transformer/sublayers.py)

- CRF
  - articles
    - [Sequence Tagging with Tensorflow](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)
    - [ADVANCED: MAKING DYNAMIC DECISIONS AND THE BI-LSTM CRF](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)
  - tensorflow impl
    - [sequence_tagging/ner_model.py](https://github.com/guillaumegenthial/sequence_tagging/blob/master/model/ner_model.py)
    - [tf_ner/main.py](https://github.com/guillaumegenthial/tf_ner/blob/master/models/chars_conv_lstm_crf/main.py)
    - [tensorflow/crf.py](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/crf/python/ops/crf.py)
  - pytorch impl
    - [allennlp/conditional_random_field.py](https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py)

- pretrained LM
  - articles
    - [Contextual String Embeddings for Sequence Labeling](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view)
    - [Semi-Supervised Sequence Modeling with Cross-View Training](https://arxiv.org/pdf/1809.08370.pdf)
    - [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)
    - [Semi-supervised sequence tagging with bidirectional language models](https://arxiv.org/pdf/1705.00108.pdf)
    - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
  - tensorflow impl
    - [bilm-tf](https://github.com/allenai/bilm-tf)
    - [BERT-NER](https://github.com/kyzhouhzau/BERT-NER)
    - [BERT-BiLSTM-CRF-NER](https://github.com/macanv/BERT-BiLSMT-CRF-NER)
  - pytorch impl
    - [flair](https://github.com/zalandoresearch/flair)
   
- tensorflow 
  - tensorflow save and restore from python/C/C++
    - [save, restore tensorflow models quick complete tutorial](https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/amp/)
    - [tensorflow-cmake](https://github.com/PatWie/tensorflow-cmake)
    - [Training a Tensorflow graph in C++ API](https://tebesu.github.io/posts/Training-a-TensorFlow-graph-in-C++-API)
    - [label_image in C++](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc)
    - [how to invoke tf.initialize_all_variables in c tensorflow](https://www.queryoverflow.gdn/query/how-to-invoke-tf-initialize-all-variables-in-c-tensorflow-27_34975884.html)
    - [TensorFlow: How to freeze a model and serve it with a python API](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc)
    - [how to read freezed graph from C++](https://stackoverrun.com/ko/q/12408779)
  - inference speed up
    - GPU
      - tensorRT
        - [install tensorRT](https://developer.nvidia.com/tensorrt)
        - [Speed up TensorFlow Inference on GPUs with TensorRT](https://medium.com/tensorflow/speed-up-tensorflow-inference-on-gpus-with-tensorrt-13b49f3db3fa)
        - [how to use tensorRT](https://hiseon.me/2018/03/28/tensorflow-tensorrt/)
        - [Speed up Inference by TensorRT](https://tsmatz.wordpress.com/2018/07/07/tensorrt-tensorflow-python-on-azure-tutorial/amp/)
        - experiments
          - [x] no meaningful difference. is it not effective for batch size 1 ?
    - CPU
      - quantizing graph
        - [tf.contrib.quantize](https://www.tensorflow.org/api_docs/python/tf/contrib/quantize)
        - [Quantizing neural network to 8-bit using Tensorflow(pdf)](https://armkeil.blob.core.windows.net/developer/developer/technologies/Machine%20learning%20on%20Arm/Tutorials/Quantizing%20neural%20networks%20to%208-it%20using%20Tensorflow/Quantizing%20neural%20networks%20to%208-bit%20using%20TensorFlow.pdf)
        - [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/pdf/1806.08342.pdf)
        - experiments
          - [x] tf.import_graph_def() error after training with tf.contrib.quantize.create_training_graph(), freezing, exporting. 
            - hmm... something messy.
      - tensorflow MKL
        - [optimizing tensorflow for cpu](https://www.tensorflow.org/performance/performance_guide#optimizing_for_cpu)
        - conda tensorflow distribution
          - [miniconda](https://conda.io/miniconda.html)
          - [tensorflow in anaconda](https://www.anaconda.com/blog/developer-blog/tensorflow-in-anaconda/)
          - [tensorflow-mkl, optimizing tensorflow for cpu](http://waslleysouza.com.br/en/2018/07/optimizing-tensorflow-for-cpu/)
        - experiments
          - [x] no meaningful improvement.
  - tensorflow summary
    - [how to manually create a tf summary](https://stackoverflow.com/questions/37902705/how-to-manually-create-a-tf-summary/37915182#37915182)
  - tensorflow backend
  ```
  - implementations of BLAS specification
    - OpenBlas, intel MKL, Eigen(more functionality, high level library in C++)
  - Nvidia GPU
    - CUDA language specification and library
    - cuDNN(more functionality, high level library)
  - tensorflow
    - GPU
      - use mainly cuDNN
      - some cuBlas, GOOGLE CUDA(customized by google)
    - CPU
      - use basically Eigen
      - support MKL, MKL-DNN
      - or Eigen with MKL-DNN backend
  ```
