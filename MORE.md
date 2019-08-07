
## Evaluation

### [experiment logs](https://github.com/dsindex/etagger/blob/master/ENG_EXPERIMENT.md)

### results
  - QRNN
    - Glove
      - setting : `experiments 14, test 8`
      - per-token(partial) f1 : 0.8892680845877263
      - per-chunk(exact)   f1 : 0.8809544851966417 (conlleval)
      - average processing time per bucket
        - 1 GPU(TITAN X(Pascal), 12196MiB)
          - restore version : 0.013028464151645457 sec
        - 32 processor CPU(multi-threading)
          - python : 0.004297458387741437 sec
          - C++ : 0.004124 sec
        - 1 CPU(single-thread)
          - python : 0.004832443533451109 sec
          - C++ : 0.004734 sec
  - Transformer
    - Glove
      - setting : `experiments 7, test 9`
      - per-token(partial) f1 : 0.9083215796897038
      - per-chunk(exact)   f1 : **0.904078014184397** (chunk_eval)
      - average processing time per bucket
        - 1 GPU(TITAN X (Pascal), 12196MiB)
          - restore version        : 0.013825567226844812 sec
          - frozen version         : 0.015376264122228799 sec
          - tensorRT(FP16) version : no meaningful difference
        - 32 processor CPU(multi-threading)
          - python : 0.017238136546748987 sec
          - C++ : 0.013 sec
        - 1 CPU(single-thread)
          - python : 0.03358284470571628 sec
          - C++ : 0.021510 sec
  - BiLSTM
    - Glove
      - setting : `experiments 9, test 1`
      - per-token(partial) f1 : 0.9152852267186738
      - per-chunk(exact)   f1 : **0.9094911075893644** (chunk_eval)
      - average processing time per bucket
        - 1 GPU(TITAN X (Pascal), 12196MiB)
          - restore version        : 0.010454932072004718 sec
          - frozen version         : 0.011339560587942018 sec
          - tensorRT(FP16) version : no meaningful difference
        - 32 processor CPU(multi-threading)
          - rnn_num_layers 2 : 0.006132203450549827 sec
          - rnn_num_layers 1
            - python
              - 0.0041805055967241884 sec
              - 0.003053264560968687  sec (`experiments 12, test 5`)
            - C++
              - 0.002735 sec
              - 0.002175 sec (`experiments 9, test 2`), 0.8800
              - 0.002783 sec (`experiments 9, test 3`), 0.8858
              - 0.004407 sec (`experiments 9, test 4`), 0.8887
              - 0.003687 sec (`experiments 9, test 5`), 0.8835
              - 0.002976 sec (`experiments 9, test 6`), 0.8782
              - 0.002855 sec (`experiments 9, test 7`), 0.8906
                - 0.002697 sec with optimizations for FMA, AVX and SSE. no meaningful difference.
              - 0.002040 sec (`experiments 12, test 5`), 0.9047
        - 1 CPU(single-thread)
          - rnn_num_layers 2 : 0.008001159379070668 sec 
          - rnn_num_layers 1
            - python
              - 0.0051817628640952506 sec
              - 0.0042755354628630235 sec (`experiments 12, test 5`)
            - C++
              - 0.003998 sec
              - 0.002853 sec (`experiments 9, test 2`)
              - 0.003474 sec (`experiments 9, test 3`)
              - 0.005118 sec (`experiments 9, test 4`)
              - 0.004139 sec (`experiments 9, test 5`)
              - 0.004133 sec (`experiments 9, test 6`)
              - 0.003334 sec (`experiments 9, test 7`)
                - 0.003078 sec with optimizations for FMA, AVX and SSE. no meaningful difference.
              - 0.002683 sec (`experiments 12, test 5`)
    - ELMo
      - setting : `experiments 8, test 2`
      - per-token(partial) f1 : 0.9322728663199756
      - per-chunk(exact)   f1 : **0.9253625751680227** (chunk_eval)
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
        - 32 processor CPU(multi-threading)      : 0.40098162731570347 sec
        - 1 CPU(single-thread)              : 0.7398052649182165 sec
    - ELMo + Glove
      - setting : `experiments 10, test 16`
      - per-token(partial) f1 : 0.9322386962382061
      - per-chunk(exact)   f1 : **0.928729526339088** (chunk_eval)
      ```
      processed 46666 tokens with 5648 phrases; found: 5657 phrases; correct: 5247.
      accuracy:  98.44%; precision:  92.75%; recall:  92.90%; FB1:  92.83
                    LOC: precision:  93.89%; recall:  94.00%; FB1:  93.95  1670
                   MISC: precision:  85.03%; recall:  83.33%; FB1:  84.17  688
                    ORG: precision:  90.17%; recall:  91.63%; FB1:  90.89  1688
                    PER: precision:  97.58%; recall:  97.22%; FB1:  97.40  1611
      ```
      - average processing time per bucket
        - 1 GPU(TITAN X (Pascal), 12196MiB) : 0.036233977567360014 sec
        - 1 GPU(Tesla V100, 32510MiB) : 0.031166194639816864 sec
    - BERT `new result, aligned wordpiece+word embeddings)`
      - BERT(large) + Glove + ELMo
        - setting : `experiments 15, test 7`
        - per-token(partial) f1 :
        - per-chunk(exact)   f1 : (chunk_eval), (conlleval)
        - average processing time per bucket
          - 1 GPU(Tesla V100)  : pass
      - BERT(large) + Glove
        - setting : `experiments 15, test 6`
        - per-token(partial) f1 : 0.9223324758054636
        - per-chunk(exact)   f1 : 0.9159886805801203(chunk_eval), 91.57(conlleval)
        - average processing time per bucket
          - 1 GPU(Tesla V100)  : pass
      - BERT(large)
        - BERT + LSTM + CRF only
        - setting : `experiments 15, test 2`
        - per-token(partial) f1 : 0.9120832058733557
        - per-chunk(exact)   f1 : 0.9015151515151516(chunk_eval), 90.14(conlleval)
        - average processing time per bucket
          - 1 GPU(Tesla V100)  : pass
    - BERT `old result, extending word embeddings for wordpieces`
      - BERT(base)
        - setting : `experiments 11, test 1`
        - per-token(partial) f1 : 0.9234725113260683
        - per-chunk(exact)   f1 : 0.9131509267431598 (chunk_eval)
        - average processing time per bucket
          - 1 GPU(Tesla V100)  : 0.026964144585057526 sec
      - BERT(base) + Glove
        - setting : experiments 11, test 2`
        - per-token(partial) f1 : 0.921535076998289
        - per-chunk(exact)   f1 : 0.9123210182075304 (chunk_eval)
        - average processing time per bucket
          - 1 GPU(Tesla V100)  : 0.029030597688838533 sec
      - BERT(large)
        - BERT + CRF only
        - setting : `experiments 11, test 15`
        - per-token(partial) f1 : 0.929012534393152
        - per-chunk(exact)   f1 : 0.9215426705498191 (chunk_eval), **92.00**(conlleval)
        - average processing time per bucket
          - 1 GPU(Tesla V100)  : pass
      - BERT(large)
        - BERT + LSTM + CRF only
        - setting : `experiments 11, test 19`
        - per-token(partial) f1 : 0.9310957309977338
        - per-chunk(exact)   f1 : 0.9240976645435245 (chunk_eval), **92.23**(conlleval)
        - average processing time per bucket
          - 1 GPU(Tesla V100)  : pass
      - BERT(large) + Glove
        - setting : `experiments 11, test 3`
        - per-token(partial) f1 : 0.9278869778869779
        - per-chunk(exact)   f1 : 0.918813634351483 (chunk_eval)
        - average processing time per bucket
          - 1 GPU(Tesla V100)  : 0.040225753178425645 sec
      - BERT(large) + Glove + Transformer
        - setting : `experiments 11, test 7`
        - per-token(partial) f1 : 0.9244949032533724
        - per-chunk(exact)   f1 : 0.9170714474962465 (chunk_eval)
        - average processing time per bucket
          - 1 GPU(Tesla V100)  : 0.05737522856032033 sec
  - BiLSTM + Transformer
    - Glove
      - setting : `experiments 7, test 10`
      - per-token(partial) f1 : 0.910979409787988
      - per-chunk(exact)   f1 : **0.9047451049567825** (chunk_eval)
  - BiLSTM + multi-head attention
    - Glove
      - setting : `experiments 6, test 7`
      - per-token(partial) f1 : 0.9157317073170732
      - per-chunk(exact)   f1 : **0.9102156238953694** (chunk_eval)

### comparision to previous research
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
    - [SOTA on named-entity-recognition-ner-on-conll-2003](https://paperswithcode.com/sota/named-entity-recognition-ner-on-conll-2003)
      - [Cloze-driven Pretraining of Self-attention Networks](https://arxiv.org/pdf/1903.07785.pdf?fbclid=IwAR2eIBWLbo0EShXvIhkMtS9OCwAipX8xKMS3GibEfP5oDwzjRv8r5WdlMtc)
        - reported F1 : 0.935
      - [GCDT: A Global Context Enhanced Deep Transition Architecture for Sequence Labeling](https://arxiv.org/pdf/1906.02437v1.pdf)
        - reported F1 : 0.9347
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

## Development note

### accuracy and loss
![](/etc/graph-2.png)

### abnormal case when using multi-head
![](/etc/graph-3.png)
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
  ![](/etc/graph-4.png)

### train, dev accuracy after applying LSTMBlockFusedCell
![](/etc/graph-5.png)

### tips for training speed up
  - filter out words(which are not in train/dev/test data) from glove840B word embeddings. but not for service.
  - use LSTMBlockFusedCell for bidirectional LSTM. this is faster than LSTMCell.
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

### tips for Transformer
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
    ![](/etc/graph-6.png)

### tips in general
  - save best model by using token-based f1. token-based f1 is slightly better than chunk-based f1
  - be careful for word lowercase when you are using glove6B embeddings. those are all lowercased.
  - feed max sentence length to session. this yields huge improvement of inference speed.
  - when it comes to using import_meta_graph(), you should run global_variable_initialzer() before restore().

### tips for BERT fine-tuning
  - it seems that the warmup and exponential decay of learing rate are worth to use.
  ![](/etc/warmup-1.png)
  ![](/etc/warmup-2.png)

## References

### general
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

### character convolution
  - articles
    - [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
    - [Implementing a sentence classification using Char level CNN & RNN](https://github.com/cuteboydot/Sentence-Classification-using-Char-CNN-and-RNN)
  - tensorflow impl
    - [cnn-text-classification-tf/text_cnn.py](https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py)
    - [lstm-char-cnn-tensorflow/LSTMTDNN.py](https://github.com/carpedm20/lstm-char-cnn-tensorflow/blob/master/models/LSTMTDNN.py)

### Transformer
  - articles
    - [Building the Mighty Transformer for Sequence Tagging in PyTorch](https://medium.com/@kolloldas/building-the-mighty-transformer-for-sequence-tagging-in-pytorch-part-i-a1815655cd8)
    - [QANET: COMBINING LOCAL CONVOLUTION WITH GLOBAL SELF-ATTENTION FOR READING COMPREHENSION](https://arxiv.org/pdf/1804.09541.pdf)
  - tensorflow impl
    - [transformer/modules.py](https://github.com/Kyubyong/transformer/blob/master/modules.py)
    - [transformer-tensorflow/attention.py](https://github.com/DongjunLee/transformer-tensorflow/blob/master/transformer/attention.py)
    - [seq2seq/pooling_encoder.py](https://github.com/google/seq2seq/blob/master/seq2seq/encoders/pooling_encoder.py)
  - pytorch impl
    - [torchnlp/sublayers.py](https://github.com/kolloldas/torchnlp/blob/master/torchnlp/modules/transformer/sublayers.py)

### CRF
  - articles
    - [Sequence Tagging with Tensorflow](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)
    - [ADVANCED: MAKING DYNAMIC DECISIONS AND THE BI-LSTM CRF](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)
  - tensorflow impl
    - [sequence_tagging/ner_model.py](https://github.com/guillaumegenthial/sequence_tagging/blob/master/model/ner_model.py)
    - [tf_ner/main.py](https://github.com/guillaumegenthial/tf_ner/blob/master/models/chars_conv_lstm_crf/main.py)
    - [tensorflow/crf.py](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/crf/python/ops/crf.py)
  - pytorch impl
    - [allennlp/conditional_random_field.py](https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py)

### pretrained LM
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
   
### tensorflow 
  - tensorflow save and restore from python/C/C++
    - [save, restore tensorflow models quick complete tutorial](https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/amp/)
    - [tensorflow-cmake](https://github.com/PatWie/tensorflow-cmake)
    - [Training a Tensorflow graph in C++ API](https://tebesu.github.io/posts/Training-a-TensorFlow-graph-in-C++-API)
    - [label_image in C++](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc)
    - [how to invoke tf.initialize_all_variables in c tensorflow](https://www.queryoverflow.gdn/query/how-to-invoke-tf-initialize-all-variables-in-c-tensorflow-27_34975884.html)
    - [TensorFlow: How to freeze a model and serve it with a python API](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc)
    - [how to read freezed graph from C++](https://stackoverrun.com/ko/q/12408779)
    - [reducing model loading time and/or memory footprint](https://www.tensorflow.org/lite/tfmobile/optimizing#reducing_model_loading_time_andor_memory_footprint)
      - convert_graphdef_memmapped_format
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
        - tf.contrib.quantize
          - [tf.contrib.quantize](https://www.tensorflow.org/api_docs/python/tf/contrib/quantize)
          - [Quantizing neural network to 8-bit using Tensorflow(pdf)](https://armkeil.blob.core.windows.net/developer/developer/technologies/Machine%20learning%20on%20Arm/Tutorials/Quantizing%20neural%20networks%20to%208-it%20using%20Tensorflow/Quantizing%20neural%20networks%20to%208-bit%20using%20TensorFlow.pdf)
          - [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/pdf/1806.08342.pdf)
          - experiments
            - [x] tf.import_graph_def() error after training with tf.contrib.quantize.create_training_graph(), freezing, exporting. 
              - hmm... something messy.
        - optimize_for_inference, quantize_graph, transform_graph
          - [tensorflow-for-mobile-poets](https://petewarden.com/2016/09/27/tensorflow-for-mobile-poets/)
          - [graph_transforms](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#optimizing-for-deployment)
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
  - tfrecord, tf.data api
    - [simple_batching](https://www.tensorflow.org/guide/datasets#simple_batching)
  - tensorflow runtime include path, library path, check if built_with_cuda enabled.
  ```
  $ python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
  $ python -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
  $ python -c "import tensorflow as tf; print(int(tf.test.is_built_with_cuda()))"
  ```
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

### etc
  - QRNN
    - [QRNN](https://arxiv.org/pdf/1611.01576.pdf?fbclid=IwAR3hreOvBGmJZe54-631X49XedcbsQoDYIRu87BcCHEBf_vMKF8FDKK_7Nw)
    - [QRNN Explained](http://mlexplained.com/2018/04/09/paper-dissected-quasi-recurrent-neural-networks-explained/?fbclid=IwAR1s0khdARsUTpvgaoqeYza4BVYPKVyAHx71OfjdCKG1qJn1nBeV3Nh9ynk)
    - [tensorflow_qrnn](https://github.com/JonathanRaiman/tensorflow_qrnn)
    - [tf.reverse_sequence](https://www.tensorflow.org/api_docs/python/tf/reverse_sequence)
    - [Even sized kernels with SAME padding in Tensorflow](https://stackoverflow.com/questions/51131821/even-sized-kernels-with-same-padding-in-tensorflow)
