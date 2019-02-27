# Copyright (C) 2018 Zhixian MA <zx@mazhixian.me>
"""A class to contruct the bottleneck for encodeing

updates
=======
[2018-01-26]: Init, summary todo.
[2018-01-28]: Add odd_flag

notes
=====
1. batch norm only at the end of a bottleneck?

references
==========
http://blog.csdn.net/helei001/article/details/52692128
"""

import numpy as np
from collections import namedtuple
import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import avg_pool2d


import utils

class Bottleneck_en():
    """
    A class to contruct the bottleneck fo encoding

    inputs
    ======
    input: tf.Tensor
        a tensor as the input of last layer
    bottle_params: list
        configure params for the bottleneck, for instance
        [(row,col,kernels),stride,batch_norm_flag,padding,active_func]
        bottle_params =
            [((1,1,64), 1,False,'SAME',tf.nn.relu),
             ((3,3,64), 1,False,'SAME',tf.nn.relu),
             ((1,1,256),1,True,'SAME',tf.nn.relu)
        ]
    is_training: tf.placeholder
        training placeholder for batch normalization
        c.f. https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
    scope: str
        Scope of this bottleneck

    Methods
    =======
    get_depth_out
    get_shortcut
    get_bottlenet
    """

    def __init__(self, inputs,
                 bottle_params=[
                     ((1, 1, 64), 1, False, 'SAME', tf.nn.relu),
                     ((3, 3, 64), 1, False, 'SAME', tf.nn.relu),
                     ((1, 1, 256), 1, True, 'SAME', tf.nn.relu)],
                 is_training=None,
                 scope=None,
                 summary_flag = False
                 ):
        """
        The initializer
        """
        self.inputs = inputs
        self.input_shape = self.inputs.get_shape()[-3:]
        self.depth_in = self.input_shape[-1]
        self.bottle_params = bottle_params
        self.is_training = is_training
        self.summary_flag = summary_flag
        if scope is None:
            self.scope = 'bottleneck' + str(np.random.randint(100))
        else:
            self.scope = scope
        # init functions
        self.get_depth_out()


    def get_depth_out(self):
        """Get the output shape of this bottleneck"""
        self.depth_out = self.bottle_params[-1][0][-1]
        self.stride = self.bottle_params[-2][1]


    def get_shortcut(self, stride, scope='shortcut'):
        """Reshape and repeat to get the shortcut of input

        Reference
        =========
        [1] TensorFlow 实战
        """
        def subsample(inputs, factor, scope):
            if factor == 1:
                return inputs
            else:
                # avg for auto encoder
                return avg_pool2d(inputs,[1,1],
                                  stride=factor,
                                  padding='SAME',
                                  scope=scope)
        if self.depth_in == self.depth_out:
            self.shortcut = subsample(self.inputs, stride, scope)
        else:
            self.shortcut = conv2d(
                inputs=self.inputs,
                num_outputs=self.depth_out,
                kernel_size=[1,1],
                stride=stride,
                padding='SAME',
                normalizer_fn=None,
                activation_fn=None,
                scope=scope)


    def get_bottlenet(self):
        """Form the network"""
        # get collections
        bottlelayer = namedtuple("bottlelayer",
                              ['kernel_shape','stride','bn_flag','padding','act_fn'])
        with tf.name_scope(self.scope):
            input_now = self.inputs
            #print("bottle_net:", type(input_now), input_now)
            for i, kernel in enumerate(self.bottle_params):
                with tf.name_scope('bottle_sub'+str('i')):
                    kernel = bottlelayer._make(kernel)
                    with tf.name_scope('conv2d'):
                        residual = conv2d(
                            inputs=input_now,
                            num_outputs=kernel.kernel_shape[-1],
                            kernel_size=kernel.kernel_shape[0:2],
                            padding=kernel.padding,
                            stride=kernel.stride,
                            )
                    if kernel.bn_flag:
                        residual = utils.get_batch_norm(residual,
                                                        self.is_training,
                                                        scope=self.scope+'_batch_norm')
                    if kernel.act_fn is not None:
                        with tf.name_scope('activate'):
                            residual = kernel.act_fn(residual)
                    input_now = residual
                    # print(i, " ", residual.get_shape())
            # add shortcut
            self.get_shortcut(self.stride,scope=self.scope+'_shortcut')
            residual = residual + self.shortcut
            if self.summary_flag:
                tf.summary.histogram('bottle_residual', residual)

            if residual.get_shape()[1] % 2 == 0:
                odd_flag = False
            else:
                odd_flag = True

        return residual, odd_flag
