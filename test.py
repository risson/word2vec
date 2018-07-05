import keras
import keras.models as m
import numpy as np
from keras.preprocessing.text import Tokenizer
import pickle

with open('tokenizer.pk', 'rb') as f:
    tokenizer = pickle.load(f)

word_index = tokenizer.word_index
index_words = {value : key for key,value in word_index.items()}

model = m.load_model('saved_model.h5')
embedding_matrix = model.get_layer('shared_embedding').get_weights()[0]


def cos_matrix_multiplication(matrix, vector):
    """
    Calculating pairwise cosine distance using matrix vector multiplication.
    """
    dotted = matrix.dot(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    matrix_vector_norms = np.multiply(matrix_norms, vector_norm)
    neighbors = np.divide(dotted, matrix_vector_norms)
    return neighbors

def l2_distance(matrix, vector):
    mx_subtract = np.subtract(matrix, vector)
    return np.linalg.norm(mx_subtract, axis=1)

def find_top_k(word, k=3):
    #wids = np.argsort(- cos_matrix_multiplication(embedding_matrix, embedding_matrix[word_index[word]]))[1:k+1]
    wids = np.argsort(l2_distance(embedding_matrix, embedding_matrix[word_index[word]]))[1:k+1]
    return [index_words[wid] for wid in wids]

words = ["fast","gift","bad","device","phone","computer"]

for word in words:
    print("{}: {}".format(word, find_top_k(word)))
