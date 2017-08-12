from __future__ import print_function
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from optparse import OptionParser
import os
import pickle
import time
import Tkinter

from data_preprocess import DATA_OUTPUT, DATA_DIR, WORD_DICTIONARY
from chatbot import LSTMModel
from batch_producer import BatchProducer, to_multiple_col, END_TOKEN
from train import BUCKETS
from configuration import CONFIG_DIR, Configuration, create_test_config


def cutoff_sentence(sent):
    new_sent = []
    for word in sent:
        if word != END_TOKEN:
            new_sent.append(word)
        else:
            break
    return new_sent


def chat_loop(config, model_path):
    print('preparing batch producer...')
    data_path = os.path.join(DATA_DIR, DATA_OUTPUT)
    dict_path = os.path.join(DATA_DIR, WORD_DICTIONARY)
    batch_producer = BatchProducer(data_path, dict_path,
        BUCKETS, config.init_index, config.max_sentence_count)

    print('creating graph...')
    graph = tf.Graph()
    with graph.as_default():
        print('building model...')
        model = LSTMModel(BUCKETS, config, training=False)

    with tf.Session(graph=graph) as sess:
        print('prepare saver for loading...')
        saver = tf.train.Saver()
        print('loading model...')
        saver.restore(sess, model_path)

        # gui
        root = Tkinter.Tk()
        input_text = Tkinter.Text(root, height=1, width=80)
        text_area = Tkinter.Text(root, height=30, width=80)
        input_text.pack(side=Tkinter.BOTTOM)
        text_area.pack(side=Tkinter.TOP)

        def key_event(event):
            if event.char and ord(event.char) == 13:
                text = input_text.get('1.0', Tkinter.END).strip()
                if len(text) <= 0:
                    return
                text_area.insert(Tkinter.END, 'You: ' + text + '\n')
                input_text.delete('1.0', Tkinter.END)

                # create data
                bucket_id = -1
                source, target, weight = batch_producer.create_pair(
                    text, '', bucket_id)
                encoder_inputs = to_multiple_col(source)
                decoder_inputs = to_multiple_col(target)
                target_weights = to_multiple_col(weight)

                _, _, out = model.step(
                    sess, encoder_inputs, decoder_inputs,
                    target_weights, 1, bucket_id, False)

                index = [np.argmax(o, axis=1)[0] for o in out]
                sent = batch_producer.convert_to_sentence(index)
                sent = cutoff_sentence(sent)
                machine_response = ' '.join(sent)
                if len(machine_response) <= 0:
                    machine_response = 'I don\'t understant you!!'
                text_area.insert(Tkinter.END,
                    'Machine: ' + machine_response + '\n')

        input_text.bind('<Key>', key_event)
        root.mainloop()


def main():
    parser = OptionParser()
    parser.add_option('-m', '--model_path', dest='model',
        default='model', help='model directory')
    parser.add_option('-c', '--config_path', dest='config',
        default=os.path.join(CONFIG_DIR, 'lstm-default.conf'),
        help='which configuration to use')
    (options, args) = parser.parse_args()

    if os.path.isfile(options.config):
        config = Configuration(options.config)
        test_config = create_test_config(config)
        print('train configure:')
        print(config)
        print('test configure: ')
        print(test_config)
        print('loading %s for prediction...' % (options.model))
        chat_loop(test_config, options.model)


if __name__ == '__main__':
    main()
