import numpy as np
import random
import re


def preprocess_corpus_file(corpus_file, lowfreq_unk_thres=0):
    """
    """
    word_to_id = {}
    id_to_word = {}
    word_count = {}
    total_cnt = 0
    with open(corpus_file) as fopen:
        for line in fopen:
            line = re.sub("[\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "",line)
            for word in line.strip().split(" "):
                if word is "":
                    continue
                word = word.lower()
                if word not in word_count:
                    word_count[word] = 0
                word_count[word] += 1
                total_cnt += 1

    # Make some low frequency words into unknown <UNK>
    word_count = dict(filter(lambda _:_[1]>=lowfreq_unk_thres, word_count.items()))
    vocab_size = len(word_count)
    word_to_id = dict(zip(sorted(list(word_count.keys())), range(vocab_size)))
    new_cnt = sum(_[1] for _ in word_count.items())
    word_to_id['<UNK>'] = vocab_size
    word_count['<UNK>'] = total_cnt - new_cnt
    id_to_word = dict((value, key) for key, value in word_to_id.items())

    return word_to_id, id_to_word, word_count, vocab_size

def negative_sample_array(word_id_dict, word_count, negative_array_size=1e8):
    """
    """
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

def corpus_reader(corpus_file, word_count, word_id_dict, context_window, random_window_size,
                  subsampling_t, subsampling_thres, negative_sample_size, negative_array):
    """
    """
    def reader():
        with open(corpus_file) as fopen:
            for line in fopen:
                line = re.sub("[\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "",line)
                line_list = line.strip().split(" ")
                word_ids = [word_id_dict.get(_.lower(), word_id_dict.get("<UNK>")) for _ in line_list]

                if random_window_size:
                    cur_window = random.randint(1, context_window)
                
                for i in range(len(word_ids)):
                    target = word_ids[i]
                    freq = word_count[target]
                    if discard_word(subsampling_t, subsampling_thres, freq):
                        continue
                    
                    context_list = []
                    context_list.append(target)
                    # generate positive sample
                    for j in range(i-cur_window, i+cur_window):
                        if j<0 or j==i:
                            continue
                        pos_context = word_ids[j]
                        context_list.append(pos_context)
                        yield ((target, pos_context),1)
                    
                    # generate negative sample
                    for _ in range(cur_window*2*negative_sample_size):
                        neg_context = negative_sample_generate(negative_array)
                        while neg_context in context_list:
                            neg_context = negative_sample_generate(negative_array)
                        yield ((target, neg_context),0)
    return reader
                        
                        

                    



def discard_word(t, threshold, freq):
    p_discard = 1 - (t/freq)**0.5
    if p_discard >= threshold:
        return True



if __name__ == "__main__":
    a,b,c,d = preprocess_corpus_file("/Users/risson/git/word2vec/testpage.txt")
    print(a)
    print(b)
    print(c)