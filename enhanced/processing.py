import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re
from bs4 import BeautifulSoup
import pickle
from tqdm import tqdm
import os
import numpy as np



def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem
    
    return words


def preprocess_data(data_train, data_test, data_valid, 
                    labels_train, labels_test, labels_valid,
                    cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""
    os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists
    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        #words_train = list(map(review_to_words, data_train))
        #words_test = list(map(review_to_words, data_test))
        words_train = [review_to_words(review) for review in tqdm(data_train)]
        words_valid = [review_to_words(review) for review in tqdm(data_valid)]
        words_test = [review_to_words(review) for review in tqdm(data_test)]
        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,words_valid=words_valid,
                              labels_train=labels_train, labels_test=labels_test, labels_valid=labels_valid)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, words_valid, labels_train, labels_test, labels_valid = (cache_data['words_train'],
                cache_data['words_test'], cache_data['words_valid'], 
                cache_data['labels_train'], cache_data['labels_test'], cache_data['labels_valid'])
    return words_train, words_test, words_valid, labels_train, labels_test, labels_valid


def build_dict(data, vocab_size = 5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""
    
    # TODO: Determine how often each word appears in `data`. Note that `data` is a list of sentences and that a
    #       sentence is a list of words.

    flatten_data = [word for review in data for word in review]
    (unique, counts) = np.unique(flatten_data, return_counts=True)   

    # word_count = {}
    # A dict storing the words that appear in the reviews along with how often they occur
    word_count = zip_word_count = {x:y for x,y in zip(unique, counts)}
    
    # TODO: Sort the words found in `data` so that sorted_words[0] is the most frequently appearing word and
    #       sorted_words[-1] is the least frequently appearing word.
    #sorted_zip_word_count = [{key: value} for (key, value) in sorted(zip_word_count.items(), key=lambda x: x[1], reverse=True)]
    #sorted_words = dict((key, val) for k in sorted_zip_word_count for key, val in k.items())
    sorted_words = list(key for (key, value) in sorted(zip_word_count.items(), key=lambda x: x[1], reverse=True))
    word_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
         word_dict[word] = idx + 2                              # 'infrequent' labels
        
    return word_dict, sorted_words, word_count


def update_dict(word_dict, words_to_remove=[]):
    updated_dict = word_dict.copy()
    for word in words_to_remove:
        updated_dict.pop(word)
        
    for token in updated_dict:
        updated_dict[token] = updated_dict[token] - len(words_to_remove)
        
    new_vocab_size = len(updated_dict) + 2
    
    return updated_dict, new_vocab_size


def convert_and_pad(word_dict, sentence, pad=500):
    NOWORD = 0 # We will use 0 to represent the 'no word' category
    INFREQ = 1 # and we use 1 to represent the infrequent words, i.e., words not appearing in word_dict
    
    working_sentence = [NOWORD] * pad
    
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ
            
    return working_sentence, min(len(sentence), pad)


def convert_and_pad_data(word_dict, data, pad=500):
    result = []
    lengths = []
    
    for sentence in tqdm(data,leave=True):
        converted, leng = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(leng)
        
    return np.array(result), np.array(lengths)