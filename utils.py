import numpy as np
import random


def preprocess_corpus_file(corpus_file, separator=" ", punct_kept=False, lowfreq_unk_thres=3, highfreq_subsampling=False,
                           subsampling_t=1e-5, subsampling_thres=0.8):
    """
    """
    with open(corpus_file) as fopen:
        
        word_to_id = {}
        id_to_word = {}
        word_count = {}
        total_cnt = 0
        for line in fopen:
            for word in line.strip().split(separator):
                if punct_kept:
                    print("Currently not keeping punctuation. This function will be built later.")
                else:
                    word = word.lower().strip().split(',')[0]
                    word = word.strip().split('.')[0]
                    word = word.strip().split('！')[0]
                    word = word.strip().split('[')[0]
                    word = word.strip().split(']')[0]
                    print(word)


def negative_sample_generate():
    """
    """

def negative_sample_shuffle():
    """
    """

def corpus_reader():
    """
    """

if __name__ == "__main__":
    preprocess_corpus_file("c:/Users/risson.yao/Documents/AI Edu/NLP/simpletest.txt")