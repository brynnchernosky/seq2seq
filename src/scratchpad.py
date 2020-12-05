import numpy as np
import tensorflow as tf

def scratchpad(model, decoder_hidden_state, encoder_output, attentive_read):
    #update probability for each state with shape 100, 14 - equation 5
    #decoder hidden state 100, 256 
    #encoder output 100, 14, 256
    #attentive read 100, 1, 256
    decoder_hidden_state = tf.expand_dims(decoder_hidden_state,1)
    #make decoder hidden state and attentive read have middle dimension 14
    stacked_decoder_state = decoder_hidden_state
    for i in range(13):
        stacked_decoder_state = tf.concat([stacked_decoder_state, decoder_hidden_state],1)
    stacked_attention = attentive_read
    for i in range(13):
        stacked_attention = tf.concat([stacked_attention, attentive_read],1)    
    prob = tf.concat([stacked_decoder_state, stacked_attention, encoder_output],2) #100, 14, 768
    prob = model.scratchpad_dense1(prob) #100, 14, 100
    prob = tf.nn.softmax(prob, 1)
    update_probs = prob #100 14
    
    #computes global update with shape 100, 256 - equation 6
    global_update = tf.concat([tf.squeeze(decoder_hidden_state),tf.squeeze(attentive_read)],1)
    global_update = model.scratchpad_dense2(global_update) 
    global_update = tf.math.tanh(global_update) #100 256

    #update encoder states, equation 4

    #update probs shape 100 14 100
    #encoder output shape 100 14 256
    #global update shape 100 256

    probs = tf.squeeze(tf.slice(update_probs,[0,0,0],[100,1,100])) #100 100
    out = tf.squeeze(tf.slice(encoder_output,[0,0,0],[100,1,256])) #100 256
    outputs = tf.matmul(probs,out) + tf.matmul((1-probs),global_update)
    outputs = tf.expand_dims(outputs, 1) #100,1,256
    for i in range(1,14):
        probs = tf.squeeze(tf.slice(update_probs,[0,i,0],[100,1,100])) #100 100
        out = tf.squeeze(tf.slice(encoder_output,[0,i,0],[100,1,256])) #100 256
        output = tf.matmul(probs,out) + tf.matmul((1-probs),global_update)
        output = tf.expand_dims(output, 1)
        outputs = tf.concat([outputs, output], 1)

    return outputs #shape 100, 14, 256
