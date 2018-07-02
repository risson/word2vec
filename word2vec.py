#----- Import Packages -----#

import os
import numpy as np
#import tensorflow as tf
#import tensorflow.contrib.eager as tfe
from collections import Counter 
import utils2
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, dot, Reshape, Dense, Merge, Activation, merge
from keras.preprocessing.text import Tokenizer
import datetime
import pickle

#tf.enable_eager_execution()

class SkipGramNS(object):
    """
    Implementing word2vec using Skip-Gram and Negative Sampling
    """

    def __init__(self, corpus_file,
                 embed_dim=100, optimizer="adam", loss="binary_crossentropy"):
        """
        corpus_file: the file that contains the corpus
        lowfreq_unk_thres: the threshold to deside whether a low frequency word should be replaced by '<UNK>'
        embed_dim: the dimension of the output word2vec
        window_size: size of the Skip_Gram window for context
        optimizer: define the method for optimization. Possible values are: "adam", "RMSprop", "momentum"...
        loss: define the loss function. Possible values are: "binary_crossentropy", "hinge"...
        """
        # tf.enable_eager_execution()
        
        lines = []
        with open("/Users/risson/git/word2vec/reviews_Electronics_5.txt") as fopen:
            for idx, line in enumerate(fopen):
                if idx > 100000:
                    break
                lines.append(line.strip())
        self._vocab_size = len(set(_ for line in lines for _ in line.split()))
        self._tokenizer = Tokenizer(num_words=self._vocab_size)
        self._tokenizer.fit_on_texts(lines)
        with open('tokenizer.pk', 'wb') as tokenizer_f:
            pickle.dump(self._tokenizer, tokenizer_f)
        self._corpus_file = corpus_file
        self._embed_dim = embed_dim
        self._vocab_size = self._tokenizer.num_words
        self._model = self.__init_model__(self._vocab_size, embed_dim, optimizer, loss)

    def __init_model__(self, vocab_size, embed_dim, optimizer, loss):
        """

        """
        target = Input(shape=(1,), name="target", dtype="int32")
        context = Input(shape=(1,), name="context", dtype="int32")
        shared_embed = Embedding(vocab_size, embed_dim, input_length=1, name="shared_embedding")
        target_embed = shared_embed(target)
        context_embed = shared_embed(context)
        merged_vec = merge([target_embed, context_embed], mode="cos", dot_axes=-1)
        reshaped_vec = Reshape((1,))(merged_vec)
        prediction = Dense(units=1, activation="sigmoid")(reshaped_vec)
        
        model = Model(inputs=[target,context], outputs=prediction)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model
    
    def save(self, path):
        self._model.save(path)

    def train(self, epoches=10, batch_size=256, negative_sample_size=2, context_window=2, random_window_size=True,
              negative_array_size=1e7, subsampling_t=1e-5,  subsampling_thres=0.8, is_shuffle=True):
        """
        epoches: number of epoches. In 1 epoch we will process the entire input data set
        batch_size: size of the batch for 1 forward feeding and backprop
        negative_sample_size:   negative sample size, which is k times bigger than positive sample size. 
                                If set as 0, then no negative samples. If set as 1, then same size as positive samples
        highfreq_subsampling: whether we want to remove words with high frequency
        subsampling_t: the t value to calculate probability of a word to be removed
        subsampling_thres: the threshold to calculate probability of a word to be removed 
        """
        neg_array = utils2.negative_sample_array(self._tokenizer, negative_array_size)
        reader = utils2.corpus_reader(self._corpus_file, self._tokenizer, context_window, random_window_size, 
                                      subsampling_t, subsampling_thres, negative_sample_size, neg_array)
        
        if is_shuffle:
            reader = utils2.shuffle(reader, batch_size*50)

        for epoch in range(epoches):
            batch_id = 0
            batch_x = [[],[]]
            batch_y = []
            loss_list = []

            for word_context, label in reader():
                batch_x[0].append(word_context[0])
                batch_x[1].append(word_context[1])
                batch_y.append(label)

                batch_id += 1

                if batch_id%(batch_size*100)==0:
                    print("Epoch {}: Batch_id - {}: Loss = {}".format(epoch, batch_id, np.mean(loss_list)))
                    loss_list = []

                if batch_id%batch_size==0:
                    X = [np.array(batch_x[0]), np.array(batch_x[1])]
                    
                    loss = self._model.train_on_batch(X, np.array(batch_y))
                    loss_list.append(loss)
                    batch_x = [[],[]]
                    batch_y = []

            print("{}: Epoch {} done".format(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"), epoch))
            self.save('saved_model.h5')
    
    def return_word_vec(self, output_file, use_id):
        """
        
        """
        word_dict = dict((_[1],_[0]) for _ in self._tokenizer.word_index.items())
        with open(output_file) as f:
            f.write("%d %d\n" % (len(word_dict), self._embed_dim))
            for idx, vec in enumerate(self._model.layers[2].get_weights()[0].tolist()):
                if use_id:
                    f.write("%d %s\n" % (idx, " ".join(str(_) for _ in vec)))
                else:
                    f.write("%s %s\n" % (word_dict[idx], " ".join(str(_) for _ in vec)))
        
if __name__ == "__main__":
    skgram = SkipGramNS("/Users/risson/git/word2vec/reviews_Electronics_5.txt")
    skgram.train()
