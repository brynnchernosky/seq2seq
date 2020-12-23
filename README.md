# seq2seq

Our partial implementation of the paper "Keeping Notes: Conditional Natural Language Generation with a Scratchpad Mechanism" https://arxiv.org/pdf/1906.05275v1.pdf using Python, TensorFlow, and Keras.

We compare the performance of a traditional RNN seq2seq model with an enhanced RNN seq2seq model for translation between French and English using Canadian hansards. The enhanced model adds an attention mechanism and a "scratchpad" layer, the equations for which are detailed in the paper.

Effectively, attentive read compute scores for encoder outputs by concatenating decoder state and encoder output states, transposing, passing through two weight matrices, taking softmax, then returning the sum of the product of the softmaxed score matrix and the encoder output states. 

The scratchpad layer computes update probability for each encoder state by concatenating the decoder state, attentive read, and encoder outputs, passing through dense layer, and taking softmax; then computes global update by concatenating decoder state and attentive read, passing through dense layer, and taking tanh; then updates encoder states by summing update probability times current encoder state with 1-update probability times global update.


