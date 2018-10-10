import tensorflow as tf
import os
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

# Location of pretrained LM.  Here we use the test fixtures.
datadir = os.path.join('embeddings')
vocab_file = os.path.join(datadir, 'elmo_vocab.txt')
options_file = os.path.join(datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
weight_file = os.path.join(datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')

# Create a Batcher to map text to character ids.
batcher = Batcher(vocab_file, 50)

# Input placeholders to the biLM.
question_character_ids = tf.placeholder('int32', shape=(None, None, 50))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(options_file, weight_file)

# Get ops to compute the LM embeddings.
question_embeddings_op = bilm(question_character_ids)

# Get an op to compute ELMo (weighted average of the internal biLM layers)
elmo_question_input = weight_layers('input', question_embeddings_op, l2_coef=0.0)
elmo_question_output = weight_layers('output', question_embeddings_op, l2_coef=0.0)

tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]

with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    question_ids = batcher.batch_sentences(tokenized_question)
    print(question_ids)

    # Compute ELMo representations (here for the input only, for simplicity).
    elmo_question_input_ = sess.run([elmo_question_input['weighted_op']],
                                     feed_dict={question_character_ids: question_ids})
    print(elmo_question_input_)

##### general usage #####
'''
1. we have 'tokenized_question' for real input texts.
2. get elmo_question_input_
3. concat glove embedding + elmo_question_input_
3. take contextual encoding(via LSTM, Transformer encoder)
4. concat contextual encoding + elmo_question_output_
'''
