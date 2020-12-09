import numpy as np
import tensorflow as tf


class Seq2Seq(tf.keras.Model):
    def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):
        super(Seq2Seq, self).__init__()

        self.french_vocab_size = french_vocab_size  # The size of the french vocab
        self.english_vocab_size = english_vocab_size  # The size of the english vocab

        self.french_window_size = french_window_size  # The french window size
        self.english_window_size = english_window_size  # The english window size

        self.batch_size = 100
        self.embedding_size = 40
        self.learning_rate = 0.01

        self.gru_encode = tf.keras.layers.GRU(256, return_sequences=True, return_state=True)
        self.gru_decode = tf.keras.layers.GRU(256, return_sequences=True, return_state=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.feed_forward1 = tf.keras.layers.Dense(256, activation='relu')
        self.feed_forward2 = tf.keras.layers.Dense(self.english_vocab_size, activation='softmax')
        self.french_embed = tf.Variable(
            tf.random.truncated_normal([self.french_vocab_size, self.embedding_size], stddev=0.01))

    @tf.function
    def call(self, encoder_input, decoder_input):
        """
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""
        print("normal model call function")
        french_embedded_inputs = tf.nn.embedding_lookup(self.french_embed, encoder_input)
        layer, final_state = self.gru_encode(french_embedded_inputs)
        layer2, final_state2 = self.gru_decode(layer, initial_state=final_state)
        dense1 = self.feed_forward1(layer2)
        dense2 = self.feed_forward2(dense1)
        return dense2

    def accuracy_function(self, prbs, labels, mask):
        """
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

        return tf.reduce_sum(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs), mask))
