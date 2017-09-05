# -*- coding: utf-8 -*-

from __future__ import print_function
import pickle
import os
import copy


CONFIG_DIR = 'config'


class Configuration(object):
    def __init__(self, config_path=None):
        # default values
        self._config_path = config_path
        self._conf = self.load(config_path)


    def load(self, config_path):
        if config_path and not config_path.endswith('.conf'):
            config_path += '.conf'
        conf = {
            'name': 'default',
            'batch_size': 10,
            'hidden_layer_count': 2,
            'hidden_size': 512,
            'init_index': 0,
            'max_sentence_count': -1,
            'max_epoch': 1000,
            'decay_epoch': 100,
            'max_grad_norm': 10,
            'learning_rate': 1.0,
            'learning_rate_decay': 1.0 / 1.001,
            'softmax_sampling': 512,
            'weight_init': 0.06
        }
        if config_path and os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                for line in f:
                    entry = line.strip().split(' = ')
                    key = entry[0]
                    try:
                        value = int(entry[1])
                    except:
                        try:
                            value = float(entry[1])
                        except:
                            value = str(entry[1])
                    conf[key] = value
        return conf


    def save(self):
        if not self._config_path.endswith('.conf'):
            self._config_path += '.conf'
        with open(self._config_path, 'w') as f:
            for key in self._conf:
                f.write(key + ' = ' + str(self._conf[key]) + '\n')


    @property
    def name(self):
        return self._conf['name']

    @name.setter
    def name(self, value):
        self._conf['name'] = value

    @property
    def batch_size(self):
        return self._conf['batch_size']

    @batch_size.setter
    def batch_size(self, value):
        self._conf['batch_size'] = value

    @property
    def hidden_layer_count(self):
        return self._conf['hidden_layer_count']

    @hidden_layer_count.setter
    def hidden_layer_count(self, value):
        self._conf['hidden_layer_count'] = value

    @property
    def hidden_size(self):
        return self._conf['hidden_size']

    @hidden_size.setter
    def hidden_size(self, value):
        self._conf['hidden_size'] = value

    @property
    def init_index(self):
        return self._conf['init_index']

    @init_index.setter
    def init_index(self, value):
        self._conf['init_index'] = value

    @property
    def max_sentence_count(self):
        return self._conf['max_sentence_count']

    @max_sentence_count.setter
    def max_sentence_count(self, value):
        self._conf['max_sentence_count'] = value

    @property
    def max_epoch(self):
        return self._conf['max_epoch']

    @max_epoch.setter
    def max_epoch(self, value):
        self._conf['max_epoch'] = value

    @property
    def decay_epoch(self):
        return self._conf['decay_epoch']

    @decay_epoch.setter
    def decay_epoch(self, value):
        self._conf['decay_epoch'] = value

    @property
    def max_grad_norm(self):
        return self._conf['max_grad_norm']

    @max_grad_norm.setter
    def max_grad_norm(self, value):
        self._conf['maX_grad_norm'] = value

    @property
    def learning_rate(self):
        return self._conf['learning_rate']

    @learning_rate.setter
    def learning_rate(self, value):
        self._conf['learning_rate'] = value

    @property
    def learning_rate_decay(self):
        return self._conf['learning_rate_decay']

    @learning_rate_decay.setter
    def learning_rate_decay(self, value):
        self._conf['learning_rate_decay'] = value

    @property
    def softmax_sampling(self):
        return self._conf['softmax_sampling']

    @softmax_sampling.setter
    def softmax_sampling(self, value):
        self._conf['softmax_sampling'] = value

    @property
    def weight_init(self):
        return self._conf['weight_init']

    @weight_init.setter
    def weight_init(self, value):
        self._conf['weight_init'] = value

    def __str__(self):
        return 'Config name: %s\n' % (self.name) + \
            'Batch size: %d\n' % (self.batch_size) + \
            'Hidden layer count: %d\n' % (self.hidden_layer_count) + \
            'Hidden size: %d\n' % (self.hidden_size) + \
            'Initial index: %d\n' % (self.init_index) + \
            'Max sentence count: %d\n' % (self.max_sentence_count) + \
            'Max epoch: %d\n' % (self.max_epoch) + \
            'Decay epoch: %d\n' % (self.decay_epoch) + \
            'Max gradient norm: %f\n' % (self.max_grad_norm) + \
            'Learning rate: %f\n' % (self.learning_rate) + \
            'Learning rate decay: %f\n' % (self.learning_rate_decay) + \
            'Softmax sampling: %d\n' % (self.softmax_sampling) + \
            'Weight init: %f\n' % (self.weight_init)


def create_config_dir():
    if not os.path.isdir(CONFIG_DIR):
        print('creating config directory...')
        os.mkdir(CONFIG_DIR)


def create_default_config():
    print('creating default config...')
    name = 'lstm-default'
    conf = Configuration(os.path.join(CONFIG_DIR, name))
    conf.name = name
    conf.save()
    print(conf)


def create_unit_test_config():
    print('creating unit test config...')
    name = 'lstm-unit-test'
    conf = Configuration(os.path.join(CONFIG_DIR, name))
    conf.name = name
    conf.batch_size = 3
    conf.hidden_layer_count = 2
    conf.hidden_size = 16
    conf.save()
    print(conf)


def create_test_config(configuration):
    new_config = copy.deepcopy(configuration)
    new_config.batch_size = 1
    return new_config


def main():
    create_config_dir()
    create_default_config()
    create_unit_test_config()


if __name__ == '__main__':
    main()
