import numpy as np
import tensorflow as tf
import sys

from enhanced_model import Seq2SeqWithAttention
from model import Seq2Seq
from preprocess import get_data, FRENCH_WINDOW_SIZE, ENGLISH_WINDOW_SIZE


def train(model, train_french, train_english, eng_padding_index):
    """
    Runs through one epoch - all training examples.
    :param model: the initialized model to use for forward and backward pass
    :param train_french: french train data (all data for training) of shape (num_sentences, 14)
    :param train_english: english train data (all data for training) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
    :return: None
    """
    cur_range = 0
    total_loss = 0
    total_acc = 0
    total_words = 0

    while cur_range + model.batch_size < len(train_french):
        with tf.GradientTape() as tape:
            model.call(train_french[cur_range: cur_range + model.batch_size],
                               train_english[cur_range: cur_range + model.batch_size, :-1])


            probs = model.call(train_french[cur_range: cur_range + model.batch_size],
                               train_english[cur_range: cur_range + model.batch_size, :-1])

            loss_mask = train_english[cur_range: cur_range + model.batch_size, 1:] != eng_padding_index

            cur_loss = model.loss_function(probs, train_english[cur_range: cur_range + model.batch_size, 1:], loss_mask)

            cur_loss /= np.count_nonzero(loss_mask)

            print(cur_loss)

        gradients = tape.gradient(cur_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        cur_range += model.batch_size


def test(model, test_french, test_english, eng_padding_index):
    """
    Runs through one epoch - all testing examples.
    :param model: the initialized model to use for forward and backward pass
    :param test_french: french test data (all data for testing) of shape (num_sentences, 14)
    :param test_english: english test data (all data for testing) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
    :returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set,
    e.g. (my_perplexity, my_accuracy)
    """

    total_loss = 0
    total_acc = 0
    total_words = 0
    cur_range = 0

    while cur_range + model.batch_size < len(test_french):
        probs = model.call(test_french[cur_range: cur_range + model.batch_size], test_english[cur_range: cur_range + model.batch_size, :-1])
        loss_mask = test_english[cur_range: cur_range + model.batch_size, 1:] != eng_padding_index

        batch_words = np.count_nonzero(loss_mask)
        total_words += batch_words

        total_loss += model.loss_function(probs, test_english[cur_range: cur_range + model.batch_size, 1:], loss_mask)
        total_acc += batch_words*(model.accuracy_function(probs, test_english[cur_range: cur_range + model.batch_size, 1:], loss_mask))

        cur_range += model.batch_size

    perplexity = np.exp(total_loss/total_words)
    accuracy = total_acc/total_words

    return perplexity, accuracy

def main():
    # if len(sys.argv) != 2 or sys.argv[1] not in {"RNN", "ENHANCED"}:
    #     print("USAGE: python main.py <Model Type>")
    #     print("<Model Type>: [RNN/ENHANCED]")
    #     exit()

    train_english, test_english, train_french, test_french, \
        english_vocab, french_vocab, eng_padding_index = get_data(
            '../data/fls.txt', '../data/els.txt', '../data/flt.txt', '../data/elt.txt')

    print("data has been preprocessed")

    model_args = (FRENCH_WINDOW_SIZE, len(french_vocab), ENGLISH_WINDOW_SIZE, len(english_vocab))

    print("running enhanced model")
    model = Seq2SeqWithAttention(*model_args)
    print(model)
    model.call(5,5,5,5)
    #
    # if sys.argv[1] == "RNN":
    #     print("running normal model")
    #     model = Seq2Seq(*model_args)
    # else:
    #     print("running enhanced model")
    #     model = Seq2SeqWithAttention(*model_args)

    train(model, train_french, train_english, eng_padding_index)



if __name__ == '__main__':
    main()
