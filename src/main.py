import numpy as np
import tensorflow as tf
import sys

from enhanced_model import Seq2SeqWithAttention
from model import Seq2Seq
from preprocess import get_data, FRENCH_WINDOW_SIZE, ENGLISH_WINDOW_SIZE

from scipy.stats import ttest_rel
import matplotlib.pyplot as plt


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

    sample_size = 15

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
    train(model,train_french,train_english, eng_padding_index)

    #
    # if sys.argv[1] == "RNN":
    #     print("running normal model")
    #     model = Seq2Seq(*model_args)
    # else:
    #     print("running enhanced model")
    #     model = Seq2SeqWithAttention(*model_args)

    #   Lists to hold normal model data and enhanced model data

    # normal_model_accuracy_data = []
    # enhanced_model_accuracy_data = []
    #
    # normal_model_perplexity_data = []
    # enhanced_model_perplexity_data = []
    #
    # for _ in range(sample_size):
    #     #   Create model each round, train and test model and append accuracy from test() accordingly
    #
    #     normal_model = model(model_args)
    #     enhanced_model = enchanced_model(model_args)
    #
    #     train(normal_model, train_french, train_english, eng_padding_index)
    #     train(enchanced_model, train_french, train_english, eng_padding_index)
    #
    #     reg_perplexity, reg_acc = test(normal_model, test_french, test_english, eng_padding_index)[0]
    #     enh_perplexity, enh_acc = test(enhanced_model, test_french, test_english, eng_padding_index)[0]
    #
    #     normal_model_accuracy_data.appened(reg_acc)
    #     enhanced_model_accuracy_data.appened(enh_acc)
    #
    #     normal_model_perplexity_data.append(reg_perplexity)
    #     enhanced_model_perplexity_data.append(enh_perplexity)
    #
    # statistical_significance_test(normal_model_data, enhanced_model_data)
    #
    # #   Generate scatter plot showing Accuracies and Perplexities of each model in different groups
    #
    # graph = plt.figure()
    # axis = graph.add_subplot()
    #
    # axis.scatter(normal_model_accuracy_data, normal_model_perplexity_data, c="red", label="Normal Model Data")
    # axis.scatter(enhanced_model_accuracy_data, enhanced_model_perplexity_data, c="blue", label="Enhanced Model Data")
    #
    # plt.show()


def statistical_significance_test(normal_model_data, enhanced_model_data):

    #   Performs a paired T test on the normal model data and enchanced model data (accuracies only)

    statistic_val, p_val = ttest_rel(normal_model_data, enhanced_model_data)

    print(p_val)

    if p_val < 0.5:
        print("Difference in Accuracies is Statistically Significant")

    else:
        print("Difference in Accuracies is not Statistically Significant")


if __name__ == '__main__':
    main()
