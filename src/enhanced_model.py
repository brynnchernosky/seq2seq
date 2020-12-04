import numpy as np
import tensorflow as tf
import attention as attention
import scratchpad as scratchpad


class Seq2SeqWithAttention(tf.keras.Model):
    def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):
        super(Seq2SeqWithAttention, self).__init__()
        self.french_vocab_size = french_vocab_size
        self.french_window_size = french_window_size
        self.english_window_size = english_window_size
        self.english_vocab_size = english_vocab_size
        self.batch_size = 100
        self.embedding_size = 40
        self.learning_rate = 0.01

        self.gru_encoder = tf.keras.layers.GRU(256, return_sequences=True, return_state=True)
        self.gru_decoder_cell = tf.keras.layers.GRUCell(256)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.dense_state_update = tf.keras.layers.Dense(256, activation='relu')
        self.feed_forward1 = tf.keras.layers.Dense(256, activation='relu')
        self.feed_forward2 = tf.keras.layers.Dense(self.english_vocab_size, activation='softmax')
        self.eng_embed = tf.Variable(
            tf.random.truncated_normal([self.english_vocab_size, self.embedding_size], stddev=0.01))
        self.french_embed = tf.Variable(
            tf.random.truncated_normal([self.french_vocab_size, self.embedding_size], stddev=0.01))

        self.attention_weights1 = tf.Variable(
            tf.random.truncated_normal([200,384000], stddev=0.01))
        self.attention_weights2 = tf.Variable(
            tf.random.truncated_normal([100, 200], stddev=0.01))

        self.scratchpad_dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.scratchpad_dense2 = tf.keras.layers.Dense(english_window_size)

    @tf.function
    def call(self, encoder_input, decoder_input):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :param decoder_input: batched ids corresponding to english sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """
        
        french_embedded_inputs = tf.nn.embedding_lookup(self.french_embed, encoder_input)
        eng_embedded_inputs = tf.nn.embedding_lookup(self.eng_embed, decoder_input)
        # h's        final state s0
        enc_outputs, enc_state = self.gru_encoder(french_embedded_inputs)
        decoder_state = enc_state
        final_output = np.zeros(len(eng_embedded_inputs))

        # in lecture he starts with the stop token as the first input
        for i in range(tf.size(eng_embedded_inputs)):
            # this needs to end if W is the stop token..? i think
            # the attentive read enc_output
            attentive_read = attention.attention_func(self, decoder_state, enc_outputs)
            # . Update si using the most recently generated output token, yi−1, and the results
            # of the attentive read (ci). ?????
            final_output[i], decoder_state = self.gru_decoder_cell.call(attentive_read, decoder_state, True)
            enc_outputs = scratchpad.scratchpad(self, decoder_state, enc_outputs, attentive_read)

    def accuracy_function(self, prbs, labels, mask):
        """
        DO NOT CHANGE

        Computes the batch accuracy
        
        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """
        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32), mask))
        return accuracy

    def loss_function(self, prbs, labels, mask):
        """
        Calculates the model cross-entropy loss after one forward pass
        Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        # Note: you can reuse this from rnn_model.

        return tf.reduce_sum(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs), mask))
