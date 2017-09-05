from __future__ import print_function
from nltk.tokenize import word_tokenize
from optparse import OptionParser
import numpy as np
import tensorflow as tf
import os
import pickle
import time
import random

from data_preprocess import DATA_OUTPUT, DATA_DIR, WORD_DICTIONARY
from data_preprocess import START_TOKEN, END_TOKEN, UNKNOWN_TOKEN
from chatbot import LSTMModel
from configuration import CONFIG_DIR, Configuration, create_test_config
from batch_producer import BatchProducer, to_multiple_col

SUMMARY_DIR = 'train-sum'
MODEL_DIR = 'model'
TEST_SOURCE = 'hi!'
TEST_TARGET = 'hello.'
#  BUCKETS = [(10, 10), (16, 16), (20, 20), (26, 26), (30, 30), (36, 36)]
BUCKETS = [(10, 10), (16, 16), (26, 26)]


def create_dir(model_name):
    print('creating model directories...')
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    return model_path


def train(model_name, saved_model, config):
    print('starting session...')
    with tf.Session() as sess:
        print('building graph for chatbot...')
        model = LSTMModel(BUCKETS, config, True)
        print('initialize variables...')
        sess.run(tf.global_variables_initializer())

        model_path = create_dir(model_name)
        check_path = os.path.join(model_path, model_name + '.ckpt')
        train_path = os.path.join(model_path, SUMMARY_DIR)
        print('output model path: %s' % (model_path))
        train_writer = tf.summary.FileWriter(train_path)

        if saved_model and os.path.isfile(saved_model):
            print('restore saved model: %s ...' % (saved_model))
            model.saver.restore(sess, saved_model)

        print('creating batch producer...')
        data_path = os.path.join(DATA_DIR, DATA_OUTPUT)
        dict_path = os.path.join(DATA_DIR, WORD_DICTIONARY)
        batch_producer = BatchProducer(data_path, dict_path,
            BUCKETS, config.init_index, config.max_sentence_count)

        step_time = 0
        for epoch in range(config.max_epoch):

            start = time.time()
            if epoch % config.decay_epoch == 0 and epoch != 0:
                # test output
                bucket_id = len(BUCKETS) - 1
                test_source, test_target, weight = batch_producer.create_pair(
                    TEST_SOURCE, TEST_TARGET, bucket_id)
                encoder_inputs = to_multiple_col(test_source)
                decoder_inputs = to_multiple_col(test_target)
                target_weights = to_multiple_col(weight)

                loss, train_summary, outputs = model.step(
                    sess, encoder_inputs, decoder_inputs,
                    target_weights, 1, bucket_id, False)
                step_time += time.time() - start
                test_prediction = [batch_producer.to_word(o) for o in outputs]
                print('%d. test loss: %f | step time: %.3f | output: %s' % (
                    epoch, loss, step_time * 1.0 / (epoch + 1),
                    ' '.join(test_prediction)))

                sess.run(model.decay_lr)
                if epoch % (config.decay_epoch * 3) == 0 and epoch != 0:
                    print('saving...')
                    model.saver.save(sess, check_path,
                        global_step=model.global_step)
                    train_writer.add_summary(train_summary, epoch)
                    print('saved.')
            else:
                # training
                bucket_id = random.choice(range(len(BUCKETS)))
                source, target, weight = batch_producer.next_batch(
                    config.batch_size, bucket_id)
                encoder_inputs = to_multiple_col(source)
                decoder_inputs = to_multiple_col(target)
                target_weights = to_multiple_col(weight)

                _, loss = model.step(sess, encoder_inputs, decoder_inputs,
                    target_weights, config.batch_size, bucket_id, True)
                if epoch % (config.decay_epoch / 10) == 0:
                    print('%d. loss: %f' % (epoch, loss))

        print('saving last trained model...')
        model.saver.save(sess, check_path,
            global_step=model.global_step)
        train_writer.add_summary(train_summary, epoch)


def main():
    parser = OptionParser()
    parser.add_option('-t', '--train', dest='train_path',
        default=os.path.join(CONFIG_DIR, 'lstm-default'),
        help='train configuration')
    parser.add_option('-s', '--saved_model', dest='saved_model',
        default=None, help='saved model for continue training')

    (options, args) = parser.parse_args()
    print('loading training configuration...')
    train_path = options.train_path
    config = Configuration(train_path)
    print('training config:')
    print(config)

    train(config.name, options.saved_model, config)


if __name__ == '__main__':
    main()
