# Copyright (C) 2018 Zhixian MA <zx@mazhixian.me>
"""
A class to generate the ResNet block composed of bottlenecks.
"""

import numpy as np
import tensorflow as tf
from Bottleneck_en import Bottleneck_en
from Bottleneck_de import Bottleneck_de


class Block():
    """
    A class to generate the ResNet block composed of bottlenecks.

    inputs
    ======
    input: tf.Tensor
        a tensor as the input of last layer
    block_params: list
        configure params for the block, for instance
        block_params = [bottle_param1, bottle_params2, bottle_params3]
        For a bottleneck,
        [(row,col,kernels),stride,batch_norm_flag,padding,active_func]
        bottle_params =
            [((1,1,64), 1,False,'SAME',tf.nn.relu),
             ((3,3,64), 1,False,'SAME',tf.nn.relu),
             ((1,1,256),1,True,'SAME',tf.nn.relu)
        ]
    is_training: tf.placeholder
        training placeholder for batch normalization
        c.f. https://www.tensorflow.org/api_docs/python/tf/contrib/
             layers/batch_norm
    scope: str
        Scope of this block

    methods
    =======
    get_block
    """
    def __init__(self, inputs,
                 block_params,
                 is_training,
                 encode_flag=True,
                 scope=None,
                 summary_flag=False,
                 odd_flags=None
                 ):
        """
        The initializer
        """
        self.inputs = inputs
        self.block_params = block_params
        self.is_training = is_training
        self.encode_flag = encode_flag
        self.summary_flag = summary_flag
        self.odd_flags = odd_flags
        if scope is None:
            self.scope = 'block' + str(np.random.randint(100))
        else:
            self.scope = scope

    def get_block(self):
        """Get the blocks"""
        with tf.name_scope(self.scope):
            input_now = self.inputs
            if self.encode_flag:
                if self.odd_flags is None:
                    odd_flags = []
                else:
                    odd_flags = self.odd_flags
                for i, bottle_params in enumerate(self.block_params):
                    # Init bottleneck
                    bottleneck = Bottleneck_en(
                        inputs=input_now,
                        bottle_params=bottle_params,
                        is_training=self.is_training,
                        scope=self.scope+'/bottleneck_en' + str(i),
                        summary_flag=self.summary_flag
                        )
                    input_now, odd = bottleneck.get_bottlenet()
                    odd_flags.append(odd)

                # Add summary
                if self.summary_flag:
                    tf.summary.histogram('block_output', input_now)
                return input_now, odd_flags
            else:
                for i, bottle_params in enumerate(self.block_params):
                    bottleneck = Bottleneck_de(
                        inputs=input_now,
                        bottle_params=bottle_params,
                        is_training=self.is_training,
                        scope=self.scope+'/bottleneck_de' + str(i),
                        summary_flag=self.summary_flag,
                        odd_flag=self.odd_flags[i]
                        )
                    input_now = bottleneck.get_bottlenet()
                    print(input_now.get_shape())
                # Add summary
                if self.summary_flag:
                    tf.summary.histogram('block_output', input_now)
                return input_now
