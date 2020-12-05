import numpy as np
import tensorflow as tf

def scratchpad(model, decoder_hidden_state, encoder_output, attentive_read):
    #compute update probability for each state, equation 5
    enc_output = tf.slice(encoder_output, [0,0,0],[np.shape(encoder_output)[0],1,np.shape(encoder_output)[2]])
    enc_output = tf.squeeze(enc_output)
    prob = tf.concat([decoder_hidden_state, tf.squeeze(attentive_read), enc_output], 1)
    prob = model.scratchpad_dense1(prob)
    prob = tf.sigmoid(prob)
    update_probs = prob
    update_probs = tf.expand_dims(update_probs,1)
    for i in range(1,14):
        enc_output = tf.slice(encoder_output, [0,i,0],[np.shape(encoder_output)[0],1,np.shape(encoder_output)[2]])
        enc_output = tf.squeeze(enc_output)
        prob = tf.concat([decoder_hidden_state, tf.squeeze(attentive_read), enc_output], 1)
        prob = model.scratchpad_dense1(prob)
        prob = tf.sigmoid(prob)
        prob = tf.expand_dims(prob,1)
        update_probs = tf.concat([update_probs,prob],1)
    
    #computes global update, equation 6
    global_update = tf.concat([decoder_hidden_state,tf.squeeze(attentive_read)],1)
    global_update = model.scratchpad_dense2(global_update) 
    global_update = tf.math.tanh(global_update)

    #update encoder states, equation 4
    #update probs shape 100, 14, 256
    #encoder output shape 100, 14, 256
    #global update shape 100 256
    probs = tf.squeeze(tf.slice(update_probs,[0,0,0],[100,1,256])) #100 256
    out = tf.transpose(tf.squeeze(tf.slice(encoder_output,[0,0,0],[100,1,256]))) #100 256
    outputs = tf.matmul(probs,out) + tf.matmul((1-probs),tf.transpose(global_update))
    outputs = tf.expand_dims(outputs, 1) #100,1,100
    for i in range(1,14):
        probs = tf.squeeze(tf.slice(update_probs,[0,i,0],[100,1,256]))
        out = tf.transpose(tf.squeeze(tf.slice(encoder_output,[0,i,0],[100,1,256])))
        output = tf.matmul(probs,out) + tf.matmul((1-probs),tf.transpose(global_update))
        output = tf.expand_dims(outputs, 1)
        print(np.shape(outputs))
        print(np.shape(output))
        tf.concat([outputs, output], 1)
        print(np.shape(outputs))

    return encoder_output #shape 100, 14, 256