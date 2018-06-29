import numpy as np
import sys
import random
import re
import string
import tensorflow as tf
from collections import Counter


def negative_sample_array(corpus_file, num_words=1e5, negative_array_size=1e8):
    """
    """
    with open(corpus_file) as fopen:
        lines = fopen.read().splitlines()
        
    corpus_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words)
    corpus_tokenizer.fit_on_texts(lines)
    text_seq = corpus_tokenizer.texts_to_sequences(lines)

    text_seq = np.concatenate(text_seq)
    word_count = dict(Counter(text_seq))
    word_count_refine = dict((_[0],_[1]**0.75) for _ in word_count.items())
    refine_cnt = sum(_[1] for _ in word_count_refine.items())
    word_freq = dict((_[0], _[1]/refine_cnt) for _ in word_count_refine.items())       

    negative_array = []

    i = 0    
    for word, freq in word_freq.items():
        if i >= negative_array_size: break
        word_num = round(freq*negative_array_size)
        for _ in range(word_num):
            if i >= negative_array_size: continue
            negative_array.append(word)
            i += 1

    return negative_array

def negative_sample_generate(negative_array):
    """
    """
    index = random.randint(0, len(negative_array))
    return negative_array[index]

def corpus_reader(corpus_file, word_count, word_id_dict, id_word_dict, context_window, random_window_size,
                  subsampling_t, subsampling_thres, negative_sample_size, negative_array):
    """
    """

if __name__ == "__main__":
    neg_array = negative_sample_array("c:/users/risson.yao/word2vec/testpage.txt", num_words=100, negative_array_size=30)
    print(neg_array)
    