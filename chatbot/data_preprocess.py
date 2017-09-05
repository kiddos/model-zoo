# -*- coding: utf-8 -*-

from __future__ import print_function
from urllib2 import urlopen
from zipfile import ZipFile
import os
import pickle
from nltk import FreqDist
from nltk.tokenize import word_tokenize
import itertools

from configuration import Configuration


VOCABULARY_SIZE = 50000
URL = 'http://www.mpi-sws.org/~cristian/data/'
TARGET = 'cornell_movie_dialogs_corpus.zip'
DATA_DIR = 'data'
DATA_OUTPUT = 'sentence.pickle'
WORD_DICTIONARY = 'dict.pickle'

START_TOKEN = '<SENT_START>'
END_TOKEN = '<SENT_END>'
UNKNOWN_TOKEN = '<UNKNOWN_TOKEN>'


def download_data():
    # create data dir if not exists
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

    # create download the data if not exists
    target_path = os.path.join(DATA_DIR, TARGET)
    if not os.path.isfile(target_path):
        with open(target_path, 'wb') as f:
            print('downloading %s' % (target_path))
            content = urlopen(URL + TARGET).read()
            f.write(content)
    else:
        print('%s already downloaded.' % (target_path))
    return target_path


def unzip_data(target_path):
    if not os.path.isfile(target_path):
        print('%s not exists' % (target_path))
        return

    print('extracting %s' % (target_path))
    with ZipFile(target_path, 'r') as zf:
        zf.extractall(DATA_DIR)


def parse_sentences():
    sentences_path = os.path.join(DATA_DIR,
        'cornell movie-dialogs corpus', 'movie_lines.txt')
    if not os.path.isfile(sentences_path):
        print('%s not extracted' % (sentences_path))
        return

    print('parsing %s...' % (sentences_path))
    sentences = []
    with open(sentences_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            entry = line.split(' +++$+++ ')
            line_number = int(entry[0][1:])
            line_text = entry[-1]
            sentences.append([line_number, line_text])
    return sentences


def tokenize(sentences, vocabulary_size,
             start_token=START_TOKEN, end_token=END_TOKEN,
             unknown_token=UNKNOWN_TOKEN):
    print('sorting sentence order...')
    sentences.sort()

    print('tokenizing...')
    tokenized_sentences = list()
    total_sentence_count = len(sentences)
    epoch = 0
    for sent in sentences:
        try:
            tokenized_sent = [start_token] + \
                word_tokenize(sent[1].lower()) + [end_token]
            tokenized_sentences.append(tokenized_sent)
            epoch += 1
            if epoch % 10000 == 0:
                print('processing %.3f%%...' %
                    (epoch * 100.0 / total_sentence_count))
        except:
            continue

    print('counting words...')
    word_count = FreqDist(itertools.chain(*tokenized_sentences))
    # need to exclude unknown words and padding
    vocab = word_count.most_common(vocabulary_size - 2)
    indexes = [x[0] for x in vocab]
    indexes.append(unknown_token)
    print('building dictionary...')
    word_indexes = dict([(w, i + 1) for i, w in enumerate(indexes)])

    # remove words not in vocabulary list as unknown token
    tokens = list()
    for sent in tokenized_sentences:
        tokens.append([w if w in word_indexes else unknown_token
            for w in sent])
    return tokens, word_indexes


def word_to_index(tokenized_sentences, word_indexes):
    print('convert tokenized word into indexes...')
    data = list()
    for sent in tokenized_sentences:
        data.append([word_indexes[w] for w in sent])
    return data


def save_data(data):
    data_target = os.path.join(DATA_DIR, DATA_OUTPUT)
    print('saving %s...' % (data_target))
    with open(data_target, 'wb') as f:
        pickle.dump(data, f)


def save_dictionary(word_indexes):
    dictionary_target = os.path.join(DATA_DIR, WORD_DICTIONARY)
    print('saving %s...' % (dictionary_target))
    with open(dictionary_target, 'wb') as f:
        pickle.dump(word_indexes, f)


def main():
    downloaded = download_data()
    unzip_data(downloaded)
    sentences = parse_sentences()

    tokenized_sentences, word_indexes = tokenize(sentences, VOCABULARY_SIZE)
    data = word_to_index(tokenized_sentences, word_indexes)
    save_data(data)
    save_dictionary(word_indexes)


if __name__ == '__main__':
    main()
