from functools import reduce
import os
import pickle
import time
import numpy as np


def split_data(X, train_sz=10000, val_sz=1000, test_sz=1000):
    train = X[:train_sz]
    val = X[train_sz:train_sz + val_sz]
    test = X[train_sz + val_sz:]

    return train, val, test


def write_frequent_words(words):
    total_frequency = reduce((lambda x, y: x + y), [freq for _, freq in words])
    with open('data/words.txt', 'w') as out:
        for word, freq in words:
            out.write("{}:{}\n".format(word, str(freq / total_frequency)))


def preprocess_text(raw_data, stop_words, top_words, col, max):
    counts = {}
    posts = []
    X = np.array([])

    # process words and build counts
    for post in raw_data:
        tokens = post[col].lower().split(' ')
        # TODO: remove punctuation
        tokens = [t for t in tokens if t not in stop_words]
        for t in tokens:
            if t not in counts:
                counts[t] = 1
            else:
                counts[t] += 1
        posts.append(tokens)
    # build dictionary
    frequent_words = [(k, counts[k]) for k in sorted(counts, key=counts.get, reverse=True)][:top_words]
    write_frequent_words(frequent_words)
    dictionary = [w for w, _ in frequent_words]

    # build feature vector for posts
    for post in posts[:max]:
        feature_vector = np.array([post.count(d) for d in dictionary])
        if X.size == 0:
            X = np.matrix(feature_vector)
        else:
            X = np.vstack((X, feature_vector))
    return X


def preprocess_boolean(raw_data, col, max):
    # is root feature
    is_root = np.zeros((len(raw_data[:max]), 1))
    for i, post in enumerate(raw_data[:max]):
        val = 1 if post[col] else 0
        is_root[i] = val
    return is_root


def preprocess_numeric(raw_data, col, max, f=None):
    # preprocess numbers with some function (i.e. normalize data)
    feature = select_col(raw_data, col, max)
    if f != None:
        feature = f(feature)
    return feature


def select_col(raw_data, col, max):
    y = np.zeros((max, 1))
    for i, post in enumerate(raw_data[:max]):
        y[i] = post[col]
    return y


def min_max(x):
    return x - x.min() / x.max() - x.min()


def preprocess(raw_data, stop_words=[], top_words=160, feature_list=[], target='popularity_score', max=0):
    """
    Process raw data.
    :param raw_data: data set
    :param stop_words: filter words
    :param top_words: number of words for the dictionary
    :param feature_list: the resulting dataset will contain only these features
    :param target:
    :param max:
    :return:
    """
    max = len(raw_data) if max == 0 else max
    features = []

    if 'text' in feature_list:
        features.append(preprocess_text(raw_data, stop_words, top_words, 'text', max))
    if 'is_root' in feature_list:
        features.append(preprocess_boolean(raw_data, 'is_root', max))
    if 'controversiality' in feature_list:
        features.append(preprocess_numeric(raw_data, 'controversiality', max))
    if 'children' in feature_list:
        features.append(preprocess_numeric(raw_data, 'children', max, min_max))
    # Add more features here

    #
    X = np.matrix([]) if len(features) == 0 else np.hstack(features)
    y = select_col(raw_data, target, max)
    return X, y


def save_data(train, val, test):
    id = int(time.time() * 100)
    directory = 'data/process/' + str(id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(train, open('{}/train-{}.dat'.format(directory, id), 'wb'))
    pickle.dump(val, open('{}/val-{}.dat'.format(directory, id), 'wb'))
    pickle.dump(test, open('{}/test-{}.dat'.format(directory, id), 'wb'))

    print("data saved in", directory)


def load_data(id):
    train = pickle.load(open('data/process/{}/train-{}.dat'.format(id, id), 'rb'))
    val = pickle.load(open('data/process/{}/val-{}.dat'.format(id, id), 'rb'))
    test = pickle.load(open('data/process/{}/test-{}.dat'.format(id, id), 'rb'))

    return train + val + test
