import numpy as np
import tensorflow as tf

def scratchpad(model, decoder_hidden_state, encoder_output, attentive_read)
    #compute update probability for each state, equation 5
    update_prob = tf.concat([decoder_hidden_state,attentive_read,encoder_output],0)
    update_prob = model.scratchpad_dense1(update_prob)
    update_prob = tf.nn.softmax()

    #computes global update, equation 6
    global_update = tf.concat([decoder_hidden_state,attentive_read],0)
    global_update = model.scratchpad_dense2(global_update) 
    global_update = tf.math.tanh(global_update)

    #update encoder states, equation 4
    encoder_output = tf.matmul(update_prob,encoder_output) + tf.matmul((1-update_prob),global_update)
