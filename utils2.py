import numpy as np
import sys
import random
import re
import string
#import tensorflow as tf
#import tensorflow.contrib.eager as tfe
from collections import Counter
from pathlib import Path
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import make_sampling_table
import gzip

def negative_sample_array(tokenizer, negative_array_size=1e7):
    """
    """
    word_count = tokenizer.word_counts
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
    random.shuffle(negative_array)
    return negative_array

def corpus_reader(corpus_file, tokenizer, context_window, random_window_size,
                  subsampling_t, subsampling_thres, negative_sample_size, negative_array):
    """
    """
    
    lines = []
    with open(corpus_file, encoding='utf-8') as fopen:
        for idx, line in enumerate(fopen):
            if idx > 10000:
                break
            lines.append(line.strip())

    text_seq = tokenizer.texts_to_sequences(lines)
    vocab_size = len(tokenizer.word_index)

    sampling_table = make_sampling_table(vocab_size, sampling_factor=subsampling_t)
    
    def reader():
        neg_array_idx = 0
        for line in text_seq:
            if random_window_size:
                cur_window = random.randint(1, context_window)
            else:
                cur_window = context_window
                
            for i in range(len(line)):
                target = line[i]
                if (1 - sampling_table[target-1]) <= subsampling_thres: continue
                
                context_list = []
                context_list.append(target)
                # generate positive sample
                j = i-cur_window
                while (j <= i+cur_window):
                    while (j<0):
                        j += 1
                    if j==i: 
                        j += 1
                        continue
                    elif j>=len(line): break
                    else:
                        pos_context = line[j]
                        context_list.append(pos_context)
                        j += 1
                        yield ((target, pos_context),1)
                    
                # generate negative sample
                for _ in range((len(context_list)-1)*negative_sample_size):
                    if neg_array_idx >= len(negative_array):
                        neg_array_idx = 0

                    neg_context = negative_array[neg_array_idx]
                    while tokenizer.word_index[neg_context] in context_list:
                        neg_array_idx += 1
                        neg_context = negative_array[neg_array_idx]

                    yield ((target, tokenizer.word_index[neg_context]),0)
    return reader

def shuffle(reader,buf_size):
    def shuffle_reader():
        buf = []
        for item in reader():
            buf.append(item)
            if len(buf) >= buf_size:
                random.shuffle(buf)
                for b in buf:
                    yield b
                buf = []
        if len(buf) > 0:
            random.shuffle(buf)
            for b in buf:
                yield b
    return shuffle_reader

def process_json(input_file, output_file):
    with open(input_file,'r') as finput:
        jsonlines = finput.read().splitlines()
    lines = [json.loads(_)['reviewText'] for _ in jsonlines]

    with open(output_file,'w') as foutput:
        for line in lines:
            foutput.write(line+'\n')

if __name__ == "__main__":
    
    # #input_json = "/Users/risson/Downloads/reviews_Electronics_5.json"
    # input_json = "C:/Users/risson.yao/word2vec/Electronics_5.json"
    # #output_file = "/Users/risson/git/word2vec/reviews_Electronics_5.txt"
    # output_file = "C:/Users/risson.yao/word2vec/reviews_Electronics_5.txt"

    # if not Path(output_file).is_file():
    #     process_json(input_json, output_file)
    #     print("done")

    # corpus_file = output_file
    # lines = []
    # with open(corpus_file) as fopen:
    #     for idx, line in enumerate(fopen):
    #         if idx > 1:
    #             break
    #         lines.append(line.strip())


    # # Quality check utils functions
    # word_list = list(set(_ for line in lines for _ in line.split()))
    # print("vocab size: {}".format(len(word_list)))

    # _tokenizer = Tokenizer()
    # _tokenizer.fit_on_texts(lines)
    # print("vocab size by Tokenizer: {}".format(_tokenizer.num_words))
    
    # neg_array = negative_sample_array(_tokenizer, negative_array_size=2000)

    # creader = corpus_reader(corpus_file, _tokenizer, 2, True, 1e-5, 0, 1, neg_array)
    # shuffle(creader, 20)
    
    # print("reader sample output:")

    # i = 0

    # for context, label in creader():
    #     if i < 3:
    #         print("{}, {}".format(context, label))
    #     i += 1
    
    # print("reader length: {}".format(i))

    output = open('wiki-en-text-2m.txt', 'w', encoding='utf-8')
    i = 0
    with open('wiki-en-text.txt','r',encoding='utf-8') as fo:
        for line in fo:
            output.write(line)
            if i%100000==0:
                print('write {} lines'.format(i))
            if i > 1500000:
                break
            i += 1

    