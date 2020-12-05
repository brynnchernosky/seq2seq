import numpy as np
import tensorflow as tf


def attention_func(model, decoder_hidden_state, encoder_output):
    context = []
    for i in range(14):
        #creates attention distribution
        enc = tf.squeeze(encoder_output[:,i:i+1,:])
        #shape 100,256
        score = tf.concat([decoder_hidden_state,enc],1)
        #shape 100 512
        score = tf.transpose(score)
        #shape 512, 100
        score = tf.matmul(model.attention_weights1,score)
        #shape 200, 100
        score = tf.matmul(model.attention_weights2,score)
        #shape 256, 100
        score = tf.nn.softmax(score)
        #shape 256, 100
        
        # produce attentive read from attention distribution
        weighted_sum = tf.matmul(score,enc)
        #shape 100, 256
        context.append(weighted_sum)
    context = tf.convert_to_tensor(context)
    #shape 14, 100, 256
    context = tf.reduce_sum(context, axis = 0)
    #shape 100, 14
    return context
