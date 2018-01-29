# Copyright (C) 2018 Zhixian MA <zx@mazhixian.me>
"""A class to contruct the bottleneck for encodeing

updates
=======
[2018-01-27]: Decoding
[2018-01-28]: Repair the odd feature map size when transposes of stride being 2

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
from tensorflow.contrib.layers import conv2d_transpose
from tensorflow.contrib.layers import conv2d

import utils

class Bottleneck_de():
    """
    A class to contruct the bottleneck fo decoding

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
    odd_flag: bool
        Saved the flag of whether shape of the output layer is odd.

    Methods
    =======
    get_depth_out
    get_shortcut
    get_bottlenet
    """

    def __init__(self, inputs,
                 bottle_params=[
                     ((1, 1, 256), 1, False, 'SAME', tf.nn.relu),
                     ((3, 3, 64), 1, False, 'SAME', tf.nn.relu),
                     ((1, 1, 64), 1, True, 'SAME', tf.nn.relu)],
                 is_training=None,
                 scope=None,
                 summary_flag=False,
                 odd_flag=False,
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
        self.odd_flag = odd_flag
        if scope is None:
            self.scope = 'bottleneck_de' + str(np.random.randint(100))
        else:
            self.scope = scope
        # init functions
        self.get_depth_out()


    def get_depth_out(self):
        """Get the output shape of this bottleneck"""
        self.depth_out = self.bottle_params[-1][0][-1]
        self.depth_middle = self.bottle_params[-2][0][-1]
        self.stride = self.bottle_params[-2][1]


    def get_shortcut(self, stride, scope='shortcut'):
        """Reshape and repeat to get the shortcut of input,
           upsampling if stride = 2

        Reference
        =========
        [1] TensorFlow 实战
        """
        def upsample(inputs, stride, scope, odd_flag=False):
            with tf.name_scope(scope):
                if stride == 1:
                    return inputs
                else:
                    # upsampling by transposed convolution
                    input_shape = self.inputs.get_shape().as_list()
                    k = tf.ones(
                        [2, 2,
                         int(input_shape[3]),
                         int(input_shape[3])],
                         )
                    output_shape=[tf.shape(self.inputs)[0], input_shape[1]*stride,
                             input_shape[2]*stride, input_shape[3]]
                    up = tf.nn.conv2d_transpose(
                            value=inputs,
                            filter = k,
                            output_shape=output_shape,
                            strides=[1, stride, stride, 1],
                            padding='SAME',
                            name='upsample')
                    if odd_flag:
                        up = up[:,0:-1,0:-1,:]
                    return up
                    

        if self.depth_in == self.depth_out:
            self.shortcut = upsample(self.inputs, stride, scope, self.odd_flag)
        else:
            self.shortcut = conv2d_transpose(
                inputs=self.inputs,
                num_outputs=self.depth_out,
                kernel_size=[1,1],
                stride=stride,
                padding='SAME',
                normalizer_fn=None,
                activation_fn=None,
                scope=scope)
            if self.odd_flag:
                self.shortcut = self.shortcut[:,0:-1,0:-1,:]
           

    def get_bottlenet(self):
        """Form the network"""
        # get collections
        bottlelayer = namedtuple("bottlelayer",
                              ['kernel_shape','stride','bn_flag','padding','act_fn'])
        with tf.name_scope(self.scope):
            input_now = self.inputs
            for i, kernel in enumerate(self.bottle_params):
                with tf.name_scope('bottle_sub'+str('i')):
                    kernel = bottlelayer._make(kernel)
                    with tf.name_scope('conv2d_transpose'):
                        residual = conv2d_transpose(
                            inputs=input_now,
                            num_outputs=kernel.kernel_shape[-1],
                            kernel_size=kernel.kernel_shape[0:2],
                            padding=kernel.padding,
                            stride=kernel.stride)
                        if kernel.stride == 2 and self.odd_flag == True:
                            residual = residual[:,0:-1,0:-1,:]
                    if kernel.bn_flag:
                        residual = utils.get_batch_norm(residual,
                                                        self.is_training,
                                                        scope=self.scope+'batch_norm')
                    if kernel.act_fn is not None:
                        with tf.name_scope('activate'):
                            residual = kernel.act_fn(residual)
                    input_now = residual
                    print(i, " ", residual.get_shape())
        # add shortcut
        self.get_shortcut(self.stride,scope=self.scope+'_shortcut')
        print("shortcut ", self.shortcut.get_shape())
        residual = residual + self.shortcut
        if self.summary_flag:
            tf.summary.histogram('bottle_residual', residual)
        
        return residual
