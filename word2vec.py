#----- Import Packages -----#

import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import utils

tf.enable_eager_execution()

print("TensorFlow Version: {}".format(tf.VERSION))
print("Using Eager mode: {}".format(tf.executing_eagerly()))

class SkipGramNS(object):
    """
    Implementing word2vec using Skip-Gram and Negative Sampling
    """

    def __init__(self, corpus_file, separator=" ", punct_kept=False, lowfreq_unk_thres=3, highfreq_subsampling=False, 
    subsampling_t=1e-5,  subsampling_thres=0.8,embed_dim=300, window_size=2, optimizer="adam", loss="binary_crossentropy"):
        """
        corpus_file: the file that contains the corpus
        separator: word separator
        punct_kept: decide whether to keep punctuation 
        lowfreq_unk_thres: the threshold to deside whether a low frequency word should be replaced by '<UNK>'
        highfreq_subsampling: whether we want to remove words with high frequency
        subsampling_t: the t value to calculate probability of a word to be removed
        subsampling_thres: the threshold to calculate probability of a word to be removed 
        embed_dim: the dimension of the output word2vec
        window_size: size of the Skip_Gram window for context
        batch_size: size of the batch for 1 forward feeding and backprop
        epoches: number of epoches. In 1 epoch we will process the entire input data set
        optimizer: define the method for optimization. Possible values are: "adam", "RMSprop", "momentum"...
        loss: define the loss function. Possible values are: "binary_crossentropy", "hinge"...
        """
        self._corpus_file = corpus_file
        self._embed_dim = embed_dim
        self._word_to_id, _id_to_word, word_count = utils.preprocess_corpus_file(corpus_file, separator, lowfreq_unk_thres, highfreq_subsampling,
                                                                                 subsampling_t, subsampling_thres)
        self._model = self.__init_model__(optimizer, loss)

    def __init_model__(self, optimizer, loss):
        """

        """
    
    def train(self, epoches=10, batch_size=512, negative_sample_size=1):
        """
        negative_sample_size:   negative sample size, which is k times bigger than positive sample size. 
                                If set as 0, then no negative samples. If set as 1, then same size as positive samples
        """
    
    def return_word_vec(self):
        """

        """

