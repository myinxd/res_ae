# Copyright (C) 2018 Zhixian MA <zx@mazhixian.me>
"""
Utilities for constructing the residual neural network
"""

import os
import numpy as np
from collections import namedtuple
import tensorflow as tf
import tensorflow.contrib.layers as layers


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        with tf.name_scope('mean'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def weight_variable(shape):
    """Initialize weights"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Initialize biases"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def get_timestamp():
    """Get time at present"""
    import time
    timestamp = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
    return timestamp


def get_batch_norm(inputs,is_training,scope=None):
    """Do batch normalization"""
    bn_out = layers.batch_norm(
        inputs=inputs,
        center=True,
        scale=True,
        is_training=is_training,
        scope=scope
        )
    return bn_out

def get_block(resnet_classic_param, encode_flag=True, is_training=None):
    """Generate block parameters according to the
       classic list like structure.

    input
    =====
    resnet_classic_param: list
        a list composed of bottleneck configuration
    encode_flag: bool
        It true, encoder blosk; if false, decoder block
    is_training: tf.placeholder
        a placeholder for batch_normalization

    output
    ======
    block_params: list
        block parameters generated for class Block
    """
    block_params = []
    bottle_conf = namedtuple(
        'bottle_conf',
        ['depth3','depth1','stride'])

    if encode_flag:
        for bottle in resnet_classic_param:
            bottle = bottle_conf._make(bottle)
            bottle_params = []
            # layer 1
            bottle_params.append(
                ((1, 1, bottle.depth1), 1, False,'SAME', tf.nn.relu))
            # layer 2
            bottle_params.append(
                ((3, 3, bottle.depth1), bottle.stride, False, 'SAME', tf.nn.relu))
            # layer 3
            bottle_params.append(
                ((1, 1, bottle.depth3), 1, True, 'SAME', tf.nn.relu))
            block_params.append(bottle_params)
    else:
        for bottle in resnet_classic_param:
            bottle = bottle_conf._make(bottle)
            bottle_params = []
            # layer 1
            bottle_params.append(
                ((1, 1, bottle.depth3), 1, False,'SAME', tf.nn.relu))
            # layer 2
            bottle_params.append(
                ((3, 3, bottle.depth1), bottle.stride, False, 'SAME', tf.nn.relu))
            # layer 3
            bottle_params.append(
                ((1, 1, bottle.depth1), 1, True, 'SAME', tf.nn.relu))
            block_params.append(bottle_params)

    return block_params
