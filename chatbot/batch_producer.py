# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import pickle
import os
import time
from nltk.tokenize import word_tokenize

from data_preprocess import DATA_DIR, DATA_OUTPUT, WORD_DICTIONARY
from data_preprocess import START_TOKEN, END_TOKEN, UNKNOWN_TOKEN


PADDING_SYMBOL = 0


def to_multiple_col(data):
    cols = []
    for i in range(data.shape[1]):
        cols.append(np.copy(data[:, i]))
    return cols


def filter_sentences(sentences, max_length, pad_start=True):
    filtered_sentences = list()
    for sent in sentences:
        if len(sent) > max_length:
            continue
        elif len(sent) <= max_length:
            filtered_sentences.append(sent)
    return filtered_sentences


class BatchProducer(object):
    def __init__(self, sentences_path, dict_path, buckets,
            init_index, max_sentence_index):
        print('loading tokenized sentences...')
        self.sentences = self._load(sentences_path)
        self.word_count = sum([len(sent) for sent in self.sentences])
        self.sentence_count = len(self.sentences)
        print('word count: %d' % (self.word_count))
        print('sentence count: %d' % (self.sentence_count))

        self.buckets = buckets
        print('buckets: %s' % (buckets))
        if max_sentence_index > 0:
            self.sentence_count = min(self.sentence_count, max_sentence_index)
        print('max sentence index: %d' % (self.sentence_count))

        print('loading vocabularies...')
        self.vocabs = self._load(dict_path)
        self.index_to_vocab = dict([[i, w] for w, i in self.vocabs.items()])
        print('vocab size: %d' % (len(self.index_to_vocab)))

        # index for next batch
        self.index = init_index


    def _load(self, path):
        with open(path, 'r') as f:
            saved = pickle.load(f)
        return saved


    def _add_padding(self, sentence, required_size, pad_start=True):
        sent = sentence
        missing = required_size - len(sent)
        if pad_start:
            sent = [PADDING_SYMBOL for i in range(missing)] + sent
        else:
            sent = sent + [PADDING_SYMBOL for i in range(missing)]
        return sent


    def next_pair(self):
        source_sent = self.sentences[self.index]
        self.index += 1
        if self.index >= self.sentence_count:
            self.index = 0
        target_sent = self.sentences[self.index]
        return source_sent, target_sent


    def next(self, bucket):
        source_size, target_size = bucket
        # filter out sentence longer than largest bucket
        source_sent, target_sent = self.next_pair()
        while len(source_sent) > source_size or  \
                len(target_sent) > target_size:
            source_sent, target_sent = self.next_pair()
        # add padding
        source = np.array(self._add_padding(source_sent, source_size),
            dtype=np.int32)
        target = np.array(self._add_padding(target_sent, target_size, False),
            dtype=np.int32)
        weight = np.zeros(shape=target.shape, dtype=np.float32)
        weight[target != PADDING_SYMBOL] = 1.0
        return source, target, weight


    def next_batch(self, batch_size, bucket_id):
        bucket = self.buckets[bucket_id]
        source = np.ndarray(shape=[batch_size, bucket[0]], dtype=np.int32)
        target = np.ndarray(shape=[batch_size, bucket[1]], dtype=np.int32)
        weight = np.ndarray(shape=[batch_size, bucket[1]], dtype=np.int32)
        for i in range(batch_size):
            s, t, w = self.next(bucket)
            source[i, :] = s
            target[i, :] = t
            weight[i, :] = w
        return source, target, weight


    def word_to_index(self, sentence):
        tokenized = [START_TOKEN] + word_tokenize(sentence) + [END_TOKEN]
        indexes = [self.vocabs[w] if w in self.vocabs else
                self.vocabs[UNKNOWN_TOKEN] for w in tokenized]
        return indexes


    def create_test_batch(self, sentence):
        indexes = self.word_to_index(sentence)
        sent_size = len(indexes)
        for ss, _ in self.buckets:
            if sent_size < ss:
                sent_size = ss
                break
        indexes = self._add_padding(indexes, sent_size)
        indexes = [indexes]
        return np.array(indexes, dtype=np.int32)


    def convert_to_sentence(self, indexes):
        sentences = []
        for idx in indexes:
            i = int(str(idx))
            sentences.append(self.index_to_vocab[i]
                if i in self.index_to_vocab else '')
        return sentences


    def create_pair(self, source, target, bucket_id):
        source_index = self.word_to_index(source)
        target_index = self.word_to_index(target)

        bucket = self.buckets[bucket_id]
        source_index = np.array(self._add_padding(
            source_index, bucket[0]), dtype=np.int32)
        target_index = np.array(self._add_padding(
            target_index, bucket[1], False), dtype=np.int32)
        weights = np.zeros(shape=target_index.shape, dtype=np.float32)
        weights[target_index != PADDING_SYMBOL] = 1.0
        source_index = source_index.reshape([1, -1]).astype(np.int32)
        target_index = target_index.reshape([1, -1]).astype(np.int32)
        weights = weights.reshape([1, -1]).astype(np.float32)
        return source_index, target_index, weights


    def to_word(self, vec):
        index = np.argmax(vec, axis=1)
        if index[0] in self.index_to_vocab:
            return self.index_to_vocab[index[0]]
        return ''


def main():
    data_path = os.path.join(DATA_DIR, DATA_OUTPUT)
    dict_path = os.path.join(DATA_DIR, WORD_DICTIONARY)
    buckets = [(10, 10), (16, 16), (20, 20)]
    batch_producer = BatchProducer(data_path, dict_path, buckets, 0, 10)

    print('single:')
    for i in range(3):
        source, target, weight = batch_producer.next(buckets[i])
        print('source: ', end='')
        print(source)
        print('target: ', end='')
        print(target)
        print('weight: ', end='')
        print(weight)

    source, target, weight = batch_producer.next_batch(10, 0)
    print('batched:')
    print('source:')
    print(source)
    print('target:')
    print(target)
    print('weight:')
    print(weight)

    test = batch_producer.create_test_batch('hello world')
    print('test data:', end='')
    print(test)

    sentence = batch_producer.convert_to_sentence(test)
    print('back to sentence: ', end='')
    print(sentence)

    source, target, weight = batch_producer.create_pair('hi', 'hello there.', 0)
    print('source: %s' % (source))
    print('target: %s' % (target))
    print('weight: %s' % (weight))


if __name__ == '__main__':
    main()
