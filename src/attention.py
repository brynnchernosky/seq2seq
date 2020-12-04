import numpy as np
import tensorflow as tf


def attention(model, decoder_hidden_state, encoder_output):
    # creates attention distribution
    decoder_hidden_state = tf.reshape(decoder_hidden_state, [-1])
    encoder_output = tf.reshape(encoder_output, [-1])
    score = tf.concat([decoder_hidden_state, encoder_output], 0)
    score = tf.transpose(score)
    print(score)

    score = model.attention_weights1(score)
    score = model.attention_weights2(score)
    distribution = tf.nn.softmax(score)

    # produce attentive read from attention distribution
    attentive_read = tf.reduce_sum(tf.multiply(distribution, encoder_output))
    return attentive_read
