import numpy as np
import tensorflow as tf
import sys
import math

from enhanced_model import Seq2SeqWithAttention
from normal_model import Seq2Seq
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

    for i in range(0, (math.floor(len(train_french)/model.batch_size)*model.batch_size), model.batch_size):
        french_training = train_french[i:i + model.batch_size]
        english_training = train_english[i:i + model.batch_size, :-1]
        labels = train_english[i:i + model.batch_size, 1:]

        with tf.GradientTape() as tape:

            probs = model.call(french_training, english_training)

            loss_mask = labels != eng_padding_index

            cur_loss = model.loss_function(probs, labels, loss_mask)

            cur_loss /= np.count_nonzero(loss_mask)

        print(cur_loss)

        gradients = tape.gradient(cur_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


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

    while cur_range + model.batch_size < (math.floor(len(test_french)/model.batch_size))*model.batch_size:
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

    sample_size = 2

    train_english, test_english, train_french, test_french, \
        english_vocab, french_vocab, eng_padding_index = get_data(
            '../data/fls.txt', '../data/els.txt', '../data/flt.txt', '../data/elt.txt')

    print("Data has been Preprocessed")

    model_args = (FRENCH_WINDOW_SIZE, len(french_vocab), ENGLISH_WINDOW_SIZE, len(english_vocab))

    #   Lists to hold normal model data and enhanced model data

    normal_model_accuracy_data = []
    enhanced_model_accuracy_data = []

    normal_model_perplexity_data = []
    enhanced_model_perplexity_data = []

    normal_model = Seq2Seq(*model_args)
    normal_checkpoint = tf.train.Checkpoint(model=normal_model)
    normal_manager = tf.train.CheckpointManager(normal_checkpoint, './normal_chkpnts', max_to_keep=3)
    # train(normal_model, train_french, train_english, eng_padding_index)
    # normal_manager.save()

    enhanced_model = Seq2SeqWithAttention(*model_args)
    enhanced_checkpoint = tf.train.Checkpoint(model=enhanced_model)
    enhanced_manager = tf.train.CheckpointManager(enhanced_checkpoint, './enhanced_chkpnts', max_to_keep=3)
    # train(enhanced_model, train_french, train_english, eng_padding_index)
    # enhanced_manager.save()
    #
    normal_checkpoint.restore(normal_manager.latest_checkpoint)
    enhanced_checkpoint.restore(enhanced_manager.latest_checkpoint)
    # train(normal_model, train_french, train_english, eng_padding_index)
    # train(enhanced_model, train_french, train_english, eng_padding_index)

    return
    for _ in range(sample_size):

        #   Create model each round, train and test model and append accuracy from test() accordingly

        reg_perplexity, reg_acc = test(normal_model, test_french, test_english, eng_padding_index)
        enh_perplexity, enh_acc = test(enhanced_model, test_french, test_english, eng_padding_index)

        print(reg_acc)
        print(reg_perplexity)
        print(enh_acc)
        print(enh_perplexity)

        normal_model_accuracy_data.append(reg_acc.numpy())
        enhanced_model_accuracy_data.append(enh_acc.numpy())

        normal_model_perplexity_data.append(reg_perplexity)
        enhanced_model_perplexity_data.append(enh_perplexity)

    print("Normal Model:")
    print("Accuracies:")
    print(normal_model_accuracy_data)
    print("Perplexities:")
    print(normal_model_perplexity_data)

    print(' ')

    print("Enhanced Model:")
    print("Accuracies:")
    print(enhanced_model_accuracy_data)
    print("Perplexities:")
    print(enhanced_model_perplexity_data)

    print(' ')

    print("Statistical Test: Accuracies")
    statistical_significance_test(normal_model_accuracy_data, enhanced_model_accuracy_data)

    print(' ')

    print("Statistical Test: Perplexities")
    statistical_significance_test(normal_model_perplexity_data, enhanced_model_perplexity_data)

    #   Generate scatter plot showing Accuracies and Perplexities of each model in different groups

    graph = plt.figure()
    axis = graph.add_subplot()
    plt.xlabel("Model Accuracies")
    plt.ylabel("Model Perplexities")
    plt.title("Enhanced & Normal Accuracies vs Perplexities")

    axis.scatter(normal_model_accuracy_data, normal_model_perplexity_data, c="red", label="Normal Model Data")
    axis.scatter(enhanced_model_accuracy_data, enhanced_model_perplexity_data, c="blue", label="Enhanced Model Data")
    axis.legend()

    plt.show()


def statistical_significance_test(normal_model_data, enhanced_model_data):

    #   Performs a paired T test on the normal model data and enchanced model data (accuracies only)

    statistic_val, p_val = ttest_rel(normal_model_data, enhanced_model_data)

    print('pvalue:', p_val)

    if p_val < 0.5:
        print("Difference is Statistically Significant")

    else:
        print("Difference is not Statistically Significant")


if __name__ == '__main__':
    main()
