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

    def __init__(self, corpus_file, lowfreq_unk_thres=3, 
                 embed_dim=300, window_size=2, optimizer="adam", loss="binary_crossentropy"):
        """
        corpus_file: the file that contains the corpus
        lowfreq_unk_thres: the threshold to deside whether a low frequency word should be replaced by '<UNK>'
        embed_dim: the dimension of the output word2vec
        window_size: size of the Skip_Gram window for context
        optimizer: define the method for optimization. Possible values are: "adam", "RMSprop", "momentum"...
        loss: define the loss function. Possible values are: "binary_crossentropy", "hinge"...
        """
        self._corpus_file = corpus_file
        self._word_to_id, _id_to_word, word_count, _vocab_size = utils.preprocess_corpus_file(corpus_file, separator, punct_kept, lowfreq_unk_thres)
        self._model = self.__init_model__(self._vocab_size, embed_dim, optimizer, loss)

    def __init_model__(self, vocab_size, embed_dim, optimizer, loss):
        """

        """
        target = tf.keras.Input(shape=(self._vocab_size,), name="target")
        context = tf.keras.Input(shape=(self._vocab_size,), name="context")

        model = MySkipGramNSModel(inputs=[target,context])
        model.compile(optimizer=optimizer, loss=loss)
        return model
    
    def train(self, epoches=10, batch_size=512, negative_sample_size=1, context_window=2, random_window_size=True,
              subsampling_t=1e-5,  subsampling_thres=0.8, negative_sample_size=1):
        """
        epoches: number of epoches. In 1 epoch we will process the entire input data set
        batch_size: size of the batch for 1 forward feeding and backprop
        negative_sample_size:   negative sample size, which is k times bigger than positive sample size. 
                                If set as 0, then no negative samples. If set as 1, then same size as positive samples
        highfreq_subsampling: whether we want to remove words with high frequency
        subsampling_t: the t value to calculate probability of a word to be removed
        subsampling_thres: the threshold to calculate probability of a word to be removed 
        """
        for epoch in range(epoches):
            for batch in range(batch_size):

    
    def return_word_vec(self):
        """

        """

class MySkipGramNSModel(tf.keras.Model):

    def __init__(self):
        self.shared_embed = tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=1, name="shared_embed")
        self.reshaped_layer = tf.keras.layers.Reshape((1,), input_shape(1,1))
        self.sigmoid_dense = tf.keras.layers.Dense(1, input_shape=(1,) activation="sigmoid")

    def call(self, inputs):
        target_embed = self.shared_embed(inputs[0])
        context_embed = self.shared_embed(inputs[1])
        merged_vector = tf.keras.layers.dot([target_embed, context_embed], axes=-1)
        reshaped_vector = self.reshaped_layer(merged_vector)
        return sigmoid_dense(reshaped_vector)