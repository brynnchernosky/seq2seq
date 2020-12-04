import numpy as np
import tensorflow as tf


def attention_func(model, decoder_hidden_state, encoder_output):
    # creates attention distribution
    decoder_hidden_state = tf.reshape(decoder_hidden_state, [1,-1])
    encoder_output = tf.reshape(encoder_output, [1,-1])
    score = tf.concat([decoder_hidden_state, encoder_output], 1)
    score = tf.transpose(score)
    print(score) #shape = 384000,1

    score = tf.matmul(model.attention_weights1,score)
    score = tf.matmul(model.attention_weights2,score)
     #shape = 100,1
    distribution = tf.nn.softmax(score)

    # produce attentive read from attention distribution
    attentive_read = tf.reduce_sum(tf.multiply(distribution, encoder_output))
    return attentive_read
