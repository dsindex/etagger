from __future__ import print_function
import tensorflow as tf

def viterbi_decode(logits, trans_params, sequence_lengths):
    """Predict by viterbi decoding

    Args:
      logits: [batch_size, fixed sequence_length, class_size]
      trans_params: [class_size, class_size]
      sequence_lengths: [batch_size]

    Returns:
      viterbi_sequences: [batch_size, variable sequence_length]
    """
    
    viterbi_sequences = []
    for logit, sequence_length in zip(logits, sequence_lengths):
        logit = logit[:sequence_length] # keep only the valid steps
        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
        viterbi_sequences += [viterbi_seq]
    return viterbi_sequences
    

