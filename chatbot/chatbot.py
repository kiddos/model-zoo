import tensorflow as tf
import numpy as np
import os
import pickle
import time

from data_preprocess import VOCABULARY_SIZE
from configuration import CONFIG_DIR, Configuration, create_test_config


class LSTMModel(object):
    def __init__(self, buckets, config, training=True):
        self.buckets = buckets
        self.hidden_layer_count = hidden_layer_count = config.hidden_layer_count
        self.hidden_size = hidden_size = config.hidden_size
        self.vocab_size = vocab_size = VOCABULARY_SIZE

        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.Variable(float(config.learning_rate),
            trainable=False, dtype=tf.float32)
        self.decay_lr = self.lr.assign(self.lr * config.learning_rate_decay)

        # softmax output
        w_t = tf.get_variable(name='softmax_w', shape=[vocab_size, hidden_size],
            dtype=tf.float32)
        b = tf.get_variable(name='softmax_b', shape=[vocab_size],
            dtype=tf.float32)
        w = tf.transpose(w_t)
        output_softmax = (w, b)

        def sample_loss(inputs, labels):
            local_labels = tf.reshape(labels, [-1, 1])
            local_inputs = tf.cast(inputs, tf.float32)
            return tf.nn.sampled_softmax_loss(w_t, b, local_inputs,
                local_labels, config.softmax_sampling, vocab_size)

        single_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        cell = single_lstm_cell
        if hidden_layer_count > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_lstm_cell] *
                hidden_layer_count)

        def seq2seq_forward(encoder_inputs, decoder_inputs, do_decode):
            return tf.nn.seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=hidden_size, output_projection=output_softmax,
                feed_previous=do_decode, dtype=tf.float32)

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        with tf.name_scope('inputs'):
            for i in range(self.buckets[-1][0]):
                self.encoder_inputs.append(tf.placeholder(dtype=tf.int32,
                    shape=[None], name='encoder%d' % i))
            for i in range(self.buckets[-1][1] + 1):
                self.decoder_inputs.append(tf.placeholder(dtype=tf.int32,
                    shape=[None], name='decoder%d' % i))
                self.target_weights.append(tf.placeholder(dtype=tf.float32,
                    shape=[None], name='weights%d' % i))

        target = [self.decoder_inputs[i + 1]
            for i in range(len(self.decoder_inputs) - 1)]
        with tf.name_scope('seq2seq'):
            if not training:
                self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, target,
                    self.target_weights, self.buckets,
                    lambda x, y: seq2seq_forward(x, y, True),
                    softmax_loss_function=sample_loss)
            else:
                self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, target,
                    self.target_weights, self.buckets,
                    lambda x, y: seq2seq_forward(x, y, False),
                    softmax_loss_function=sample_loss)
        #  # summary cost
        for i, loss in enumerate(self.losses):
            tf.summary.scalar('loss.%d' % (i), loss)

        with tf.name_scope('outputs'):
            for b in range(len(self.buckets)):
                self.outputs[b] = [tf.matmul(output, output_softmax[0]) +
                    output_softmax[1] for output in self.outputs[b]]

        if training:
            with tf.name_scope('optimizer'):
                train_vars = tf.trainable_variables()
                self.grad_norm = []
                self.updates = []
                self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
                for b in range(len(self.buckets)):
                    gradients = tf.gradients(self.losses[b], train_vars)
                    clipped_gradients, norm = tf.clip_by_global_norm(
                        gradients, config.max_grad_norm)
                    self.grad_norm.append(norm)
                    self.updates.append(self.optimizer.apply_gradients(
                        zip(clipped_gradients, train_vars),
                        global_step=self.global_step))
        # save
        self.saver = tf.train.Saver(tf.global_variables())
        self.merged = tf.summary.merge_all()


    def step(self, sess, encoder_inputs, decoder_inputs,
            target_weights, batch_size, bucket_id, training):
        encoder_size, decoder_size = self.buckets[bucket_id]
        assert(encoder_size == len(encoder_inputs))
        assert(decoder_size == len(decoder_inputs))
        assert(decoder_size == len(target_weights))
        feed_dict = {}
        for l in range(encoder_size):
            feed_dict[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            feed_dict[self.decoder_inputs[l].name] = decoder_inputs[l]
            feed_dict[self.target_weights[l].name] = target_weights[l]

        last_target = self.decoder_inputs[decoder_size].name
        feed_dict[last_target] = np.zeros([batch_size], dtype=np.int32)

        if training:
            output_feed = [self.updates[bucket_id],
                           self.grad_norm[bucket_id],
                           self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id], self.merged]
            for l in xrange(decoder_size):
                output_feed.append(self.outputs[bucket_id][l])
        outputs = sess.run(output_feed, feed_dict)
        if training:
            return outputs[1], outputs[2]
        else:
            return outputs[0], outputs[1], outputs[2:]


def main():
    buckets = [(10, 10), (20, 20)]
    with tf.variable_scope('unit-test'):
        unit_test_config = Configuration(
            os.path.join(CONFIG_DIR, 'lstm-unit-test'))
        LSTMModel(buckets, unit_test_config)
    with tf.variable_scope('test'):
        test_config = create_test_config(unit_test_config)
        LSTMModel(buckets, test_config)
    print('model built.')


if __name__ == '__main__':
    main()
