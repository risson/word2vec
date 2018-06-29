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
        self._embed_dim = embed_dim
        self._word_to_id, _id_to_word, _word_count, _vocab_size = utils.build_corpus_dicts(corpus_file, separator, punct_kept, lowfreq_unk_thres)
        self._model = self.__init_model__(self._vocab_size, embed_dim, optimizer, loss)

    def __init_model__(self, vocab_size, embed_dim, optimizer, loss):
        """

        """
        target = tf.keras.Input(shape=(self._vocab_size,), name="target")
        context = tf.keras.Input(shape=(self._vocab_size,), name="context")

        shared_embed = tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=1, name="shared_embedding")

        target_embed = shared_embed(target)
        context_embed = shared_embed(context)
        merged_vec = tf.keras.layers.dot([target_embed, context_embed], axes=-1)
        reshaped_vec = tf.keras.layers.Reshape((1,), input_shape=(1,1))(merged_vec)
        prediction = tf.keras.layers.Dense(units=1, input_shape=(1,), activation="sigmoid")(reshaped_vec)
        
        model = tf.keras.Model(inputs=[target,context], outputs=prediction)
        model.compile(optimizer=optimizer, loss=loss)
        return model
    
    def train(self, epoches=10, batch_size=512, negative_sample_size=1, context_window=2, random_window_size=True,
              negative_array_size=1e8, subsampling_t=1e-5,  subsampling_thres=0.8, negative_sample_size=1, is_shuffle=True):
        """
        epoches: number of epoches. In 1 epoch we will process the entire input data set
        batch_size: size of the batch for 1 forward feeding and backprop
        negative_sample_size:   negative sample size, which is k times bigger than positive sample size. 
                                If set as 0, then no negative samples. If set as 1, then same size as positive samples
        highfreq_subsampling: whether we want to remove words with high frequency
        subsampling_t: the t value to calculate probability of a word to be removed
        subsampling_thres: the threshold to calculate probability of a word to be removed 
        """
        neg_array = utils.negative_sample_array(_word_to_id, _word_count, negative_array_size)
        reader = utils.corpus_reader(self._corpus_file, _word_count, _word_to_id, _id_to_word, context_window, random_window_size, subsampling_t,
                               subsampling_thres, negative_sample_size, neg_array)
        
        if is_shuffle:
            reader = utils.shuffle(reader, batch_size*50)
        
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

                if batch_id%batch_size*1000==0:
                    print("Epoch {}: Batch_id - {}: Loss = {}".format(epoch, batch_id, np.mean(loss_list)))
                    loss_list = []

                if batch_id%batch_size==0:
                    X = [np.array(batch_x[0]), np.array(batch_x[1])]
                    loss = self._model.train_on_batch(X, batch_y)
                    loss_list.append(loss)

                    batch_x = [[],[]]
                    batch_y = []

            print("{}: Epoch {} done".format(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"), 10))
        
    
    def return_word_vec(self, output_file):
        """
        
        """
        with open(output_file) as f:
            f.write("%d %d\n" % (len(self._word_to_id), self._embed_dim))
            for idx, vec in enumerate(self._model.layers[2].get_weights()[0].tolist()):
                f.write("%s %s\n" % (self._id_to_word[idx], " ".join(str(_) for _ in vec)))
        
