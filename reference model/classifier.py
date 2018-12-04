from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2 as cv
import os
import math

import input

var_map = {
    'conv2d/': 'conv2d/',
    'conv2d_1/': 'conv2d_1/'
}

tf.logging.set_verbosity(tf.logging.INFO)

class Model:
    def __init__(self):
        self.classifier = tf.estimator.Estimator(model_fn=self.cnn_model_fn, model_dir="punctuation")

    def cnn_model_fn(self, features, labels, mode):
        input_layer = tf.reshape(features, [-1, 32, 32, 1], name='inputs')

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=5,
            padding="same",
            trainable=True
        )

        activation = tf.nn.relu(conv1)

        conv2 = tf.layers.conv2d(
            inputs=activation,
            filters=64,
            kernel_size=5,
            padding="same",
            trainable=True
        )

        activation = tf.nn.relu(conv2)
        flat = tf.reshape(activation, [-1, 32 * 32 * 64])
        dense_1 = self.dense_dropconnect(flat, mode)

        logits = tf.layers.dense(
            inputs=dense_1,
            units=32
        )

        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits)
        }

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions["probabilities"])

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def dense_dropconnect(self, input, mode):
        keep_rate = 0.5
        dense_kernel = tf.get_variable('dense_1/kernel', [65536, 300], dtype=tf.float32, trainable=True)
        dense_kernel = tf.layers.dropout(dense_kernel, rate=1-keep_rate, training = mode == tf.estimator.ModeKeys.TRAIN) * keep_rate
        dense_bias = tf.get_variable('dense_1/bias', [300], dtype=tf.float32, trainable=True)
        dense = tf.matmul(input, dense_kernel)
        dense = tf.add(dense, dense_bias)
        activation = tf.nn.relu(dense)
        return activation

    def kernel_regularizer(self, weights):
        regularization = 0.001
        result = tf.reduce_sum(tf.square(weights)) * regularization
        return result

    def train(self, iterations, eval_interval):
        for _ in range(iterations):
            self.classifier.train(input_fn=input.train_input_fn, steps=eval_interval)
            self.classifier.evaluate(input_fn=input.eval_input_fn, steps=math.ceil(input.dev_size / input.batch_size))

def main(unused_argv):
    model.train(2000, 100)

model = Model()
tf.app.run(main)