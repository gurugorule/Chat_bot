import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

#nltk.download('punkt')
stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stemming(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence , all_words):
    tokenized_sentence = [stemming(words) for words in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index ,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[index] = 1.0
    return bag



